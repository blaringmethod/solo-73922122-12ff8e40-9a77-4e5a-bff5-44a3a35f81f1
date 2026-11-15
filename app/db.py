"""SQLite 持久化：以“发布批次”为最小可见单位。

* 每个 (meter_id, local_date) 至多有一个已发布批次；
* 成功重传在单事务内删除旧批次全部读数并写入新批次（完整替换）；
* 映射失败的上传不执行任何写库操作，此前发布的该日数据原样保留；
* 查询只返回同一完整批次的 UTC 读数。

并发设计（多个 API 实例/进程共用同一 SQLite 文件）：

* 连接使用 WAL 日志模式：写事务不阻塞读事务，读事务看到的是事务
  开始时刻的一致性快照，替换提交瞬间也不会读到“半新半旧”；
* 发布事务用 ``BEGIN IMMEDIATE`` 开启：进入事务即取得数据库写锁，
  “读旧批次 → 删旧批次 → 插新批次”全过程在写锁保护下串行执行。
  两个合法重传不再同时基于同一旧批次插入而撞 UNIQUE 键，而是按
  提交先后排队，后到者替换先到者，两次都成功；
* ``busy_timeout`` 让锁竞争时等待而非立即报 “database is locked”；
* 查询把“读批次头 + 读读数”放进同一个读事务：只可能返回替换前或
  替换后的完整批次，不会出现批次号/行数与读数分属两代的中间态；
* 发布事务任一步失败都会整体回滚，最后一次成功发布的数据不受影响。
"""

from __future__ import annotations

import os
import sqlite3
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from .tzmap import AssignedRow

UTC = ZoneInfo("UTC")

# 锁等待上限：并发发布时后到事务在此时间内排队，而不是立即失败
_BUSY_TIMEOUT_MS = 30_000

_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS batches (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    meter_id      TEXT    NOT NULL,
    local_date    TEXT    NOT NULL,
    published_at  TEXT    NOT NULL,
    row_count     INTEGER NOT NULL,
    UNIQUE (meter_id, local_date)
);

CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    INTEGER NOT NULL
        REFERENCES batches(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    local_time  TEXT    NOT NULL,
    utc_time    TEXT    NOT NULL,
    reading     REAL    NOT NULL,
    row_status  TEXT    NOT NULL,
    UNIQUE (batch_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_readings_batch
    ON readings(batch_id, utc_time);
"""


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def get_db_path() -> str:
    return os.environ.get("HEATMETER_DB", "/data/heatmeter.db")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """打开连接。

    ``isolation_level=None`` 关闭驱动层的隐式事务：事务边界完全由本模块
    显式的 BEGIN/COMMIT/ROLLBACK 控制（发布用 BEGIN IMMEDIATE，查询用
    只读事务），避免“SELECT 在自动提交、写入才开事务”造成的快照错位。
    """
    path = db_path or get_db_path()
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(
        path,
        timeout=_BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL：读写互不阻塞，读事务获得一致性快照；该设置持久化在库文件中，
    # 已处于 WAL 时重复执行为空操作
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()


def publish_batch(
    conn: sqlite3.Connection,
    meter_id: str,
    local_date: date,
    assigned: list[AssignedRow],
) -> dict[str, Any]:
    """单事务完整替换该 (meter, date) 的已发布批次，返回新旧批次信息。

    ``BEGIN IMMEDIATE`` 在事务开始即取得写锁：并发重传在此串行排队，
    每个事务读到的都是上一事务提交后的最新批次，于是两个合法重传
    按提交顺序先后完整替换（后者替换前者），不会因
    UNIQUE(meter_id, local_date) 冲突而失败。任一步失败整体回滚，
    最后一次成功发布的数据原样保留。
    """
    date_str = local_date.isoformat()
    published_at = _utc_now_iso()

    conn.execute("BEGIN IMMEDIATE")
    try:
        old = conn.execute(
            "SELECT id, published_at, row_count FROM batches "
            "WHERE meter_id = ? AND local_date = ?",
            (meter_id, date_str),
        ).fetchone()

        # UNIQUE(meter_id, local_date) 要求旧批次行先消失；
        # 读数随外键级联删除，这里显式删除以便不依赖级联行为
        if old is not None:
            conn.execute("DELETE FROM readings WHERE batch_id = ?", (old["id"],))
            conn.execute("DELETE FROM batches WHERE id = ?", (old["id"],))

        cur = conn.execute(
            "INSERT INTO batches (meter_id, local_date, published_at, row_count) "
            "VALUES (?, ?, ?, ?)",
            (meter_id, date_str, published_at, len(assigned)),
        )
        new_batch_id = cur.lastrowid

        conn.executemany(
            "INSERT INTO readings (batch_id, seq, local_time, utc_time, reading, row_status) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    new_batch_id,
                    r.seq,
                    r.local_time.isoformat(),
                    r.utc_time.isoformat(),
                    r.reading,
                    r.status.value,
                )
                for r in assigned
            ],
        )

        conn.commit()
    except Exception:
        # 任一步失败整体回滚：最后一次成功发布的数据不受影响
        conn.rollback()
        raise

    return {
        "batch_id": new_batch_id,
        "replaced_batch_id": old["id"] if old else None,
        "previous_published_at": old["published_at"] if old else None,
        "published_at": published_at,
        "row_count": len(assigned),
    }


def get_published_batch(
    conn: sqlite3.Connection, meter_id: str, local_date: date
) -> dict[str, Any] | None:
    """返回该日已发布批次及其全部读数；无发布返回 None。

    两次 SELECT 包在同一个读事务中：读到的是同一快照，只可能返回
    替换前或替换后的完整批次，不会出现“批次头是旧批次、读数已被
    替换事务删除”之类的撕裂结果。WAL 下该读事务不阻塞并发写入。
    """
    conn.execute("BEGIN")
    try:
        batch = conn.execute(
            "SELECT id, meter_id, local_date, published_at, row_count "
            "FROM batches WHERE meter_id = ? AND local_date = ?",
            (meter_id, local_date.isoformat()),
        ).fetchone()
        if batch is None:
            conn.commit()
            return None

        rows = conn.execute(
            "SELECT seq, local_time, utc_time, reading, row_status "
            "FROM readings WHERE batch_id = ? ORDER BY utc_time ASC, seq ASC",
            (batch["id"],),
        ).fetchall()
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "batch_id": batch["id"],
        "meter_id": batch["meter_id"],
        "local_date": batch["local_date"],
        "published_at": batch["published_at"],
        "row_count": batch["row_count"],
        "readings": [dict(r) for r in rows],
    }


def list_published_dates(conn: sqlite3.Connection, meter_id: str) -> list[dict[str, Any]]:
    # 单条 SELECT 自身即是一个一致性快照，无需额外事务
    rows = conn.execute(
        "SELECT id, local_date, published_at, row_count FROM batches "
        "WHERE meter_id = ? ORDER BY local_date ASC",
        (meter_id,),
    ).fetchall()
    return [dict(r) for r in rows]
