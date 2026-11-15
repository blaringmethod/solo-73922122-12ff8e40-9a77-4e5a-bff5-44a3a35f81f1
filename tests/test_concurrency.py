"""并发发布与查询的一致性回归测试。

模拟“两个共用同一 SQLite 文件的 API 实例”：每个线程使用独立连接，
对同一 (meter_id, local_date) 并发执行整批替换发布与查询。

覆盖：
* 两个合法重传都成功，按提交顺序完整替换（批次链严格首尾相接）；
* 替换进行中的查询只读到替换前或替换后的完整批次；
* 发布中途失败整体回滚，最后一次成功发布的数据原样保留；
* API 层并发重传 + 并发被拒上传：合法请求全部成功，被拒请求不写库。
"""

from __future__ import annotations

import io
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from app.db import connect, get_published_batch, init_db, publish_batch
from app.tzmap import AssignedRow, RowStatus

UTC = ZoneInfo("UTC")
METER = "M-CONC"
DAY = date(2026, 6, 15)


def _assigned(row_count: int, base_reading: float) -> list[AssignedRow]:
    """构造 row_count 行分钟级、UTC 严格递增的已映射读数。"""
    return [
        AssignedRow(
            seq=i + 1,
            local_time=datetime(2026, 6, 15) + timedelta(minutes=i),
            reading=base_reading + i,
            status=RowStatus.NORMAL,
            utc_time=datetime(2026, 6, 15, 4, 0, tzinfo=UTC) + timedelta(minutes=i),
        )
        for i in range(row_count)
    ]


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "race.db")
    conn = connect(path)
    try:
        init_db(conn)
    finally:
        conn.close()
    return path


def test_wal_and_busy_timeout_enabled(db_path):
    """并发前提：连接处于 WAL 且设置了锁等待超时。"""
    conn = connect(db_path)
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] > 0
    finally:
        conn.close()


def test_concurrent_publishers_replace_in_commit_order(db_path):
    """两个并发重传方都成功；批次按提交顺序形成严格替换链。"""
    seed_conn = connect(db_path)
    try:
        seed = publish_batch(seed_conn, METER, DAY, _assigned(2, 0.0))
    finally:
        seed_conn.close()

    rounds = 15
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    results: list[dict] = []
    errors: list[Exception] = []

    def worker(row_count: int):
        try:
            for i in range(rounds):
                # 每轮都对齐起跑，最大化写-写竞争
                barrier.wait(timeout=10)
                conn = connect(db_path)
                try:
                    info = publish_batch(
                        conn, METER, DAY, _assigned(row_count, 100.0 + i)
                    )
                finally:
                    conn.close()
                with lock:
                    results.append(info)
        except Exception as exc:  # pragma: no cover - 失败即断言
            with lock:
                errors.append(exc)
            raise

    threads = [
        threading.Thread(target=worker, args=(3,)),
        threading.Thread(target=worker, args=(5,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "并发发布未在限定时间内完成（疑似死锁）"

    assert not errors, f"合法重传不应失败：{errors}"
    assert len(results) == 2 * rounds

    # 提交顺序链：按 batch_id 排序后，每个批次的 replaced_batch_id
    # 必须恰好等于前一个提交批次的 id —— 没有丢批、没有基于同一
    # 旧批次的“双写撞 UNIQUE”
    chain = sorted(results, key=lambda r: r["batch_id"])
    ids = [r["batch_id"] for r in chain]
    assert len(set(ids)) == len(ids)
    prev_id = seed["batch_id"]
    for info in chain:
        assert info["replaced_batch_id"] == prev_id
        prev_id = info["batch_id"]

    # 最终库中只剩最后提交者的一个完整批次
    conn = connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
        final = get_published_batch(conn, METER, DAY)
    finally:
        conn.close()
    assert final is not None
    assert final["batch_id"] == chain[-1]["batch_id"]
    assert final["row_count"] in (3, 5)
    assert len(final["readings"]) == final["row_count"]


def test_reads_return_complete_batch_during_replacement(db_path):
    """替换进行中，查询只能读到替换前或替换后的完整批次。"""
    conn = connect(db_path)
    try:
        publish_batch(conn, METER, DAY, _assigned(3, 0.0))
    finally:
        conn.close()

    stop = threading.Event()
    read_count = 0
    inconsistencies: list[str] = []
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(100):
                conn = connect(db_path)
                try:
                    publish_batch(
                        conn, METER, DAY, _assigned(3 if i % 2 else 5, float(i))
                    )
                finally:
                    conn.close()
                time.sleep(0.001)  # 让替换窗口持续存在
        finally:
            stop.set()

    def reader():
        nonlocal read_count
        try:
            while not stop.is_set():
                conn = connect(db_path)
                try:
                    batch = get_published_batch(conn, METER, DAY)
                finally:
                    conn.close()
                read_count += 1
                if batch is None:
                    inconsistencies.append("已发布批次在替换期间消失")
                    continue
                seqs = [r["seq"] for r in batch["readings"]]
                if len(seqs) != batch["row_count"]:
                    inconsistencies.append(
                        f"批次头 row_count={batch['row_count']} 但读数 {len(seqs)} 行"
                    )
                elif seqs != list(range(1, batch["row_count"] + 1)):
                    inconsistencies.append(f"读数 seq 不完整：{seqs}")
                utcs = [r["utc_time"] for r in batch["readings"]]
                if utcs != sorted(utcs) or len(set(utcs)) != len(utcs):
                    inconsistencies.append("读数 UTC 非严格递增")
        except Exception as exc:  # pragma: no cover - 失败即断言
            errors.append(exc)
            raise

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    r.start()
    w.start()
    w.join(timeout=60)
    r.join(timeout=60)
    assert not w.is_alive() and not r.is_alive(), "读写线程未在限定时间内结束"

    assert not errors, f"查询不应抛错：{errors}"
    assert read_count > 0
    assert not inconsistencies, inconsistencies[:5]


def test_failed_publish_rolls_back_preserving_last_good_batch(db_path):
    """发布事务中途失败整体回滚，最后一次成功发布的数据原样保留。"""
    conn = connect(db_path)
    try:
        good = publish_batch(conn, METER, DAY, _assigned(2, 7.0))

        # 构造会在 readings 唯一键上中途失败的批次：seq 重复
        broken = _assigned(3, 0.0)
        broken[2] = AssignedRow(
            seq=2,  # 与上一行重复 -> UNIQUE(batch_id, seq) 冲突
            local_time=broken[2].local_time,
            reading=broken[2].reading,
            status=broken[2].status,
            utc_time=broken[2].utc_time,
        )
        with pytest.raises(sqlite3.IntegrityError):
            publish_batch(conn, METER, DAY, broken)

        # 同一连接回滚后仍可正常使用，且旧批次完整无缺
        kept = get_published_batch(conn, METER, DAY)
        assert kept is not None
        assert kept["batch_id"] == good["batch_id"]
        assert kept["row_count"] == 2
        assert [r["reading"] for r in kept["readings"]] == [7.0, 8.0]
        assert conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 2
    finally:
        conn.close()


# ---------------------------------------------------------------- API 层并发


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("HEATMETER_DB", str(db_file))
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def _upload(client, meter: str, d: date, csv_text: str):
    return client.post(
        f"/meters/{meter}/dates/{d.isoformat()}/batches",
        files={"file": ("readings.csv", io.BytesIO(csv_text.encode()), "text/csv")},
    )


def test_api_concurrent_reuploads_and_rejections(client):
    """API 层：并发合法重传全部成功；并发被拒上传不破坏已发布数据。"""
    d = date(2026, 6, 15)
    seed = _upload(client, METER, d, "seq,local_time,reading\n1,00:00,1\n")
    assert seed.status_code == 200

    csv_3 = "seq,local_time,reading\n1,00:00,1\n2,00:01,2\n3,00:02,3\n"
    csv_5 = (
        "seq,local_time,reading\n1,00:00,1\n2,00:01,2\n3,00:02,3\n"
        "4,00:03,4\n5,00:04,5\n"
    )
    # 本地时间倒流 -> 无解 -> 422，不得写库
    csv_bad = "seq,local_time,reading\n1,10:00,1\n2,09:59,2\n"

    statuses: list[int] = []
    reject_payloads: list[dict] = []
    lock = threading.Lock()
    barrier = threading.Barrier(3)

    def do_valid(text: str):
        for _ in range(10):
            barrier.wait(timeout=10)
            r = _upload(client, METER, d, text)
            with lock:
                statuses.append(r.status_code)

    def do_rejected():
        for _ in range(10):
            barrier.wait(timeout=10)
            r = _upload(client, METER, d, csv_bad)
            with lock:
                statuses.append(r.status_code)
                reject_payloads.append(r.json())

    with ThreadPoolExecutor(3) as ex:
        futures = [
            ex.submit(do_valid, csv_3),
            ex.submit(do_valid, csv_5),
            ex.submit(do_rejected),
        ]
        for f in futures:
            f.result(timeout=60)

    # 20 个合法重传全部成功，10 个非法上传全部 422
    assert statuses.count(200) == 20
    assert statuses.count(422) == 10
    # 被拒上传始终报告保留了此前批次，且未产生新数据
    for payload in reject_payloads:
        detail = payload["detail"]
        assert detail["error_code"] == "NO_UNIQUE_MAPPING_NO_SOLUTION"
        assert detail["retained_previous_batch"] is True
        assert detail["retained_batch"]["batch_id"] >= 1

    # 最终查询：完整批次，行数与读数一致
    q = client.get(f"/meters/{METER}/dates/{d.isoformat()}/readings")
    assert q.status_code == 200
    body = q.json()
    assert body["row_count"] in (3, 5)
    assert len(body["readings"]) == body["row_count"]
    utcs = [r["utc_time"] for r in body["readings"]]
    assert utcs == sorted(utcs) and len(set(utcs)) == len(utcs)
