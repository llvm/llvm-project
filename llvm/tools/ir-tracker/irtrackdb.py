# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""SQLite helpers for llvm/tools/ir-tracker."""

from __future__ import annotations

import os
import sqlite3
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence

T_FILES = "ir_tracker_files"
T_INSTR = "ir_tracker_instructions"
T_META = "ir_tracker_meta"
T_PASSES = "ir_tracker_passes"
SCHEMA_VERSION = 2
VALID_KINDS = {"ir", "mir", "all"}


def open_db_readonly(path: str) -> Optional[sqlite3.Connection]:
    if not path:
        print("ir-tracker: empty database path", file=sys.stderr)
        return None
    if not os.path.isfile(path):
        print(f"ir-tracker: database not found: {path}", file=sys.stderr)
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def open_db_write(path: str) -> Optional[sqlite3.Connection]:
    if not path:
        print("ir-tracker: empty database path", file=sys.stderr)
        return None
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def init_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        f"""
        PRAGMA foreign_keys = ON;
        CREATE TABLE {T_META} (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE {T_FILES} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          path TEXT NOT NULL UNIQUE
        );
        CREATE TABLE {T_PASSES} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          seq INTEGER NOT NULL,
          kind TEXT NOT NULL,
          phase TEXT NOT NULL,
          pass_class TEXT NOT NULL,
          ir_unit TEXT NOT NULL
        );
        CREATE UNIQUE INDEX ir_tracker_idx_passes_seq
          ON {T_PASSES}(seq);
        CREATE TABLE {T_INSTR} (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pass_id INTEGER NOT NULL REFERENCES {T_PASSES}(id),
          function TEXT NOT NULL,
          basicblock TEXT NOT NULL,
          inst_seq INTEGER NOT NULL,
          opcode TEXT NOT NULL,
          inst_text TEXT NOT NULL,
          file_id INTEGER NOT NULL REFERENCES {T_FILES}(id),
          line INTEGER NOT NULL,
          col INTEGER NOT NULL
        );
        CREATE INDEX ir_tracker_idx_instr_file_loc
          ON {T_INSTR}(file_id, line, col);
        CREATE INDEX ir_tracker_idx_instr_pass
          ON {T_INSTR}(pass_id);
        """
    )
    con.execute(
        f"INSERT INTO {T_META}(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


def _get_or_create_file_id(
    con: sqlite3.Connection, cache: Dict[str, int], path: str
) -> int:
    cached = cache.get(path)
    if cached is not None:
        return cached
    con.execute(f"INSERT OR IGNORE INTO {T_FILES}(path) VALUES(?)", (path,))
    row = con.execute(f"SELECT id FROM {T_FILES} WHERE path = ?", (path,)).fetchone()
    assert row is not None
    file_id = int(row["id"])
    cache[path] = file_id
    return file_id


def _insert_pass(
    con: sqlite3.Connection,
    seq: int,
    kind: str,
    phase: str,
    pass_name: str,
    ir_unit: str,
) -> int:
    return int(
        con.execute(
            f"INSERT INTO {T_PASSES}(seq, kind, phase, pass_class, ir_unit) "
            f"VALUES(?, ?, ?, ?, ?)",
            (seq, kind, phase, pass_name, ir_unit),
        ).lastrowid
    )


def _insert_inst(
    con: sqlite3.Connection,
    file_cache: Dict[str, int],
    current_pass_id: int,
    file_path: str,
    line_s: int,
    col_s: int,
    func: str,
    bb: str,
    inst_seq_s: int,
    opcode: str,
    inst_text: str,
) -> None:
    file_id = _get_or_create_file_id(con, file_cache, file_path)
    con.execute(
        f"INSERT INTO {T_INSTR}("
        "pass_id, function, basicblock, inst_seq, opcode, inst_text, "
        "file_id, line, col) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            current_pass_id,
            func,
            bb,
            inst_seq_s,
            opcode,
            inst_text,
            file_id,
            line_s,
            col_s,
        ),
    )


def _parse_int(field: str, value: str, line_no: int) -> int:
    try:
        return int(value)
    except ValueError as err:
        print(
            f"ir-tracker: invalid {field} at line {line_no}: {value!r}",
            file=sys.stderr,
        )
        raise ValueError(f"invalid {field}") from err


def _build_db_from_tsv(con: sqlite3.Connection, input_path: str) -> tuple[int, int]:
    file_cache: Dict[str, int] = {}
    tracker_locs: Dict[str, tuple[str, int, int]] = {}
    current_pass_id: Optional[int] = None
    n_passes = 0
    n_rows = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line:
                continue

            tag = line[0]
            if tag == "P":
                parts = line.split("\t")
                if len(parts) == 5:
                    seq_s, kind, phase, pass_name, ir_unit = (
                        parts[1],
                        "ir",
                        parts[2],
                        parts[3],
                        parts[4],
                    )
                elif len(parts) == 6:
                    seq_s, kind, phase, pass_name, ir_unit = parts[1:]
                else:
                    print(
                        f"ir-tracker: malformed pass row at line {line_no}",
                        file=sys.stderr,
                    )
                    raise ValueError("malformed pass row")
                if kind not in {"ir", "mir"}:
                    print(
                        f"ir-tracker: invalid pass kind at line {line_no}: {kind!r}",
                        file=sys.stderr,
                    )
                    raise ValueError("invalid pass kind")
                current_pass_id = _insert_pass(
                    con,
                    _parse_int("pass sequence", seq_s, line_no),
                    kind,
                    phase,
                    pass_name,
                    ir_unit,
                )
                n_passes += 1
                continue

            if tag == "T":
                parts = line.split("\t")
                if len(parts) != 5:
                    print(
                        f"ir-tracker: malformed tracker-id row at line {line_no}",
                        file=sys.stderr,
                    )
                    raise ValueError("malformed tracker-id row")
                tracker_locs[parts[1]] = (
                    parts[2],
                    _parse_int("source line", parts[3], line_no),
                    _parse_int("source column", parts[4], line_no),
                )
                continue

            if tag == "I":
                if current_pass_id is None:
                    print(
                        f"ir-tracker: instruction row before first pass record at line {line_no}",
                        file=sys.stderr,
                    )
                    raise ValueError("instruction row before first pass record")

                parts = line.split("\t", 6)
                if len(parts) != 7:
                    print(
                        f"ir-tracker: malformed instruction row at line {line_no}",
                        file=sys.stderr,
                    )
                    raise ValueError("malformed instruction row")

                tracker_id = parts[5]
                loc = tracker_locs.get(tracker_id)
                if loc is None:
                    print(
                        f"ir-tracker: unknown tracker id at line {line_no}: {tracker_id}",
                        file=sys.stderr,
                    )
                    raise ValueError("unknown tracker id")

                file_path, src_line, src_col = loc
                _insert_inst(
                    con,
                    file_cache,
                    current_pass_id,
                    file_path,
                    src_line,
                    src_col,
                    parts[1],
                    parts[2],
                    _parse_int("instruction sequence", parts[3], line_no),
                    parts[4],
                    parts[6],
                )
                n_rows += 1
                continue

            print(
                f"ir-tracker: unknown TSV row kind at line {line_no}: {tag!r}",
                file=sys.stderr,
            )
            raise ValueError("unknown TSV row kind")
    return n_passes, n_rows


def build_db(input_path: str, db_path: str) -> int:
    if not input_path:
        print("ir-tracker: empty input path", file=sys.stderr)
        return 1
    if not os.path.isfile(input_path):
        print(f"ir-tracker: input not found: {input_path}", file=sys.stderr)
        return 1

    con = open_db_write(db_path)
    if not con:
        return 1

    try:
        init_schema(con)
        con.commit()
        con.execute("BEGIN IMMEDIATE")
        n_passes, n_rows = _build_db_from_tsv(con, input_path)
        con.commit()
    except (sqlite3.Error, ValueError) as err:
        print(f"ir-tracker: sqlite error while building db: {err}", file=sys.stderr)
        return 1
    finally:
        con.close()

    print(
        f"built {db_path} from {input_path}: {n_passes} pass snapshots, {n_rows} instruction rows"
    )
    return 0


def get_schema_version(con: sqlite3.Connection) -> int:
    row = con.execute(
        f"SELECT value FROM {T_META} WHERE key = 'schema_version'"
    ).fetchone()
    if not row or row["value"] is None:
        return -1
    try:
        return int(row["value"])
    except ValueError:
        return -1


def resolve_file_ids(con: sqlite3.Connection, file_pat: str) -> List[int]:
    needle = file_pat.lower()
    ids: List[int] = []
    for row in con.execute(f"SELECT id, path FROM {T_FILES}"):
        path = (row["path"] or "").lower()
        if needle in path or path.endswith(needle):
            ids.append(int(row["id"]))
    return ids


def _check_kind(kind: str) -> bool:
    if kind not in VALID_KINDS:
        print("ir-tracker: --kind must be one of ir, mir, all", file=sys.stderr)
        return False
    return True


def _kind_clause(kind: str, table_alias: str = "p") -> tuple[str, List[object]]:
    if kind == "all":
        return "", []
    return f" AND {table_alias}.kind = ?", [kind]


def run_passes(con: sqlite3.Connection, kind: str) -> int:
    if not _check_kind(kind):
        return 1
    where_sql, params = _kind_clause(kind)
    if where_sql:
        where_sql = "WHERE" + where_sql[4:]
    rows = con.execute(
        f"SELECT id, seq, kind, phase, pass_class, ir_unit FROM {T_PASSES} "
        f"{where_sql} ORDER BY seq",
        params,
    ).fetchall()
    for row in rows:
        prefix = f"{int(row['seq']):5d}  id={int(row['id']):<6}  "
        if row["kind"] != "ir":
            prefix += f"[{row['kind']}]  "
        print(f"{prefix}{row['phase']}  '{row['pass_class']}'  on '{row['ir_unit']}'")
    print(f"total passes recorded: {len(rows)}")
    return 0


def _parse_line(line_s: str) -> Optional[int]:
    try:
        line = int(line_s, 0)
    except ValueError:
        return None
    return line if line > 0 else None


def _filter_clause(
    file_ids: Sequence[int], line: int, trace_col: Optional[int], trace_opcode: str
) -> tuple[str, List[object]]:
    in_clause = ",".join("?" * len(file_ids))
    sql = f"i.file_id IN ({in_clause}) AND i.line = ?"
    params: List[object] = [*file_ids, line]
    if trace_col is not None:
        sql += " AND i.col = ?"
        params.append(trace_col)
    if trace_opcode:
        sql += " AND i.opcode = ?"
        params.append(trace_opcode)
    return sql, params


def run_trace(
    con: sqlite3.Connection,
    file_pat: str,
    line_s: str,
    trace_col: Optional[int],
    trace_opcode: str,
    kind: str,
) -> int:
    if get_schema_version(con) < 1:
        print("ir-tracker: unsupported schema version", file=sys.stderr)
        return 1
    if not _check_kind(kind):
        return 1

    file_ids = resolve_file_ids(con, file_pat)
    if not file_ids:
        print("ir-tracker: no matching file rows", file=sys.stderr)
        return 1

    line = _parse_line(line_s)
    if line is None:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1

    where_sql, params = _filter_clause(file_ids, line, trace_col, trace_opcode)
    kind_sql, kind_params = _kind_clause(kind)

    row = con.execute(
        f"SELECT MAX(p.seq) AS max_seq "
        f"FROM {T_INSTR} i JOIN {T_PASSES} p ON i.pass_id = p.id "
        f"WHERE {where_sql}{kind_sql}",
        [*params, *kind_params],
    ).fetchone()
    if not row or row["max_seq"] is None:
        print("ir-tracker: no matching instructions found", file=sys.stderr)
        return 1

    max_seq = int(row["max_seq"])
    count_row = con.execute(
        f"SELECT COUNT(*) AS c "
        f"FROM {T_INSTR} i JOIN {T_PASSES} p ON i.pass_id = p.id "
        f"WHERE p.seq = ? AND {where_sql}{kind_sql}",
        [max_seq, *params, *kind_params],
    ).fetchone()
    print(
        f"Matches at final pass (seq={max_seq}): {int(count_row['c'])} "
        f"instruction(s)"
    )

    first_row = con.execute(
        f"SELECT p.seq, p.kind, p.pass_class, p.ir_unit, COUNT(*) AS c "
        f"FROM {T_INSTR} i JOIN {T_PASSES} p ON i.pass_id = p.id "
        f"WHERE {where_sql}{kind_sql} GROUP BY p.id ORDER BY p.seq ASC LIMIT 1",
        [*params, *kind_params],
    ).fetchone()
    if first_row:
        pass_text = f"{first_row['pass_class']} on {first_row['ir_unit']}"
        if first_row["kind"] != "ir":
            pass_text = f"[{first_row['kind']}] {pass_text}"
        print(
            f"First pass with any matching instruction: seq={int(first_row['seq'])} "
            f"{pass_text} ({int(first_row['c'])} row(s))"
        )
    return 0


class ShowInstRow(NamedTuple):
    seq: int
    kind: str
    pass_class: str
    ir_unit: str
    function: str
    basicblock: str
    inst_text: str


def _print_group(rows: Sequence[ShowInstRow]) -> None:
    if not rows:
        return
    head = rows[0]
    kind_text = "" if head.kind == "ir" else f" [{head.kind}]"
    print(f"seq={head.seq}{kind_text} '{head.pass_class}' on '{head.ir_unit}'")
    current_func = ""
    current_bb = ""
    for row in rows:
        if row.function != current_func or row.basicblock != current_bb:
            print(f"  function {row.function}, block {row.basicblock}:")
            current_func = row.function
            current_bb = row.basicblock
        print(f"    {row.inst_text}")


def run_show(
    con: sqlite3.Connection,
    file_pat: str,
    line_s: str,
    trace_col: Optional[int],
    trace_opcode: str,
    seq: int,
    show_all_passes: bool,
    kind: str,
) -> int:
    if get_schema_version(con) < 1:
        print("ir-tracker: unsupported schema version", file=sys.stderr)
        return 1
    if not _check_kind(kind):
        return 1
    if show_all_passes and seq >= 0:
        print(
            "ir-tracker: --all-passes and --seq are mutually exclusive", file=sys.stderr
        )
        return 1

    file_ids = resolve_file_ids(con, file_pat)
    if not file_ids:
        print("ir-tracker: no matching file rows", file=sys.stderr)
        return 1

    line = _parse_line(line_s)
    if line is None:
        print("ir-tracker: invalid --line", file=sys.stderr)
        return 1

    where_sql, params = _filter_clause(file_ids, line, trace_col, trace_opcode)
    kind_sql, kind_params = _kind_clause(kind)
    seq_sql = ""
    if seq >= 0:
        seq_sql = " AND p.seq = ?"
        kind_params = [*kind_params, seq]

    query = (
        f"SELECT p.seq, p.kind, p.pass_class, p.ir_unit, i.function, i.basicblock, "
        f"i.inst_seq, i.inst_text "
        f"FROM {T_INSTR} i JOIN {T_PASSES} p ON i.pass_id = p.id "
        f"WHERE {where_sql}{kind_sql}{seq_sql} "
        f"ORDER BY p.seq ASC, i.function ASC, i.basicblock ASC, i.inst_seq ASC"
    )
    rows = [
        ShowInstRow(
            int(row["seq"]),
            row["kind"] or "",
            row["pass_class"] or "",
            row["ir_unit"] or "",
            row["function"] or "",
            row["basicblock"] or "",
            row["inst_text"] or "",
        )
        for row in con.execute(query, [*params, *kind_params])
    ]
    if not rows:
        print("ir-tracker: no matching instructions found", file=sys.stderr)
        return 1

    by_seq: Dict[int, List[ShowInstRow]] = {}
    for row in rows:
        by_seq.setdefault(row.seq, []).append(row)

    last_fp = None
    for current_seq in sorted(by_seq):
        group = by_seq[current_seq]
        fp = "\n".join(row.inst_text for row in group)
        if seq < 0 and not show_all_passes and fp == last_fp:
            continue
        _print_group(group)
        last_fp = fp
    return 0


def run_sql(con: sqlite3.Connection, sql: str) -> int:
    try:
        cur = con.execute(sql)
    except sqlite3.Error as err:
        print(f"ir-tracker: prepare(sql): {err}", file=sys.stderr)
        return 1

    while True:
        row = cur.fetchone()
        if row is None:
            break
        print("(" + ", ".join("None" if v is None else str(v) for v in row) + ")")
    return 0
