# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HTML report generator for ir-tracker SQLite databases.

Generates a static, three-panel site organized by function:

  * ``index.html`` — function list grouped by file, with row counts.
  * One ``<safe>.html`` per function containing:

      - Left panel: full function list (links to other function pages).
      - Middle panel: the *initial* IR snapshot of the function (the smallest
        recorded ``seq``). Each instruction is clickable.
      - Right panel: when an instruction is clicked, the panel shows the
        full pass-by-pass history of all instructions sharing that source
        location, deduplicated when text is unchanged.

The page is fully static; per-function history is embedded inline as JSON.
"""

from __future__ import annotations

import html
import json
import os
import re
import sqlite3
import sys
import zlib
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import irtrackdb


def _loc_color(loc: str) -> str:
    """Map a ``data-loc`` string to a stable pastel ``hsl(...)`` color.

    Hue is derived from a CRC32 of the loc so the same source location
    always paints to the same color across both panels and across runs.
    Saturation/lightness are kept low so the color sits visually behind
    the text and does not fight the ``selected`` / ``linked`` highlight.
    """
    h = zlib.crc32(loc.encode("utf-8")) % 360
    return f"hsl({h}, 70%, 90%)"


_STYLE_CSS = """\
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; height: 100%; font-family: -apple-system, Segoe UI, sans-serif; color: #222; }
header { padding: 6px 12px; background: #223; color: #fff; font-size: 13px; }
header a { color: #cdf; text-decoration: none; }
.layout { display: flex; height: calc(100vh - 30px); }
.panel { overflow: auto; padding: 8px 12px; }
.panel h3 { font-size: 11px; text-transform: uppercase; color: #666; margin: 0 0 6px 0; letter-spacing: .5px; }
#funcs { width: 18%; border-right: 1px solid #ccc; background: #f7f7fb; font-size: 12px; }
#funcs ul { list-style: none; padding-left: 6px; margin: 4px 0 12px 0; }
#funcs li { margin: 1px 0; }
#funcs a { text-decoration: none; color: #224; }
#funcs a.current { font-weight: bold; color: #b00; }
#funcs .file { font-weight: bold; margin-top: 6px; color: #555; }
#filter { width: 95%; padding: 3px; margin-top: 4px; }
#initial, #final { width: 27%; border-right: 1px solid #ccc; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
#history { width: 28%; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.bb { color: #553; font-weight: bold; margin: 8px 0 2px 0; }
.inst { white-space: pre; padding: 1px 4px; cursor: pointer; border-radius: 3px; }
.inst:hover { background: #eef !important; }
.inst.linked { background: #fff1c4 !important; }
.inst.selected { background: #ffe080 !important; outline: 1px solid #caa040; }
.inst.no-loc { cursor: default; color: #888; }
.inst.no-loc:hover { background: transparent; }
.inst.dead { color: #999; text-decoration: line-through; }
.final-text { white-space: pre; margin: 0; font-size: 12px; color: #111; }
.signature { white-space: pre-wrap; word-break: break-all; padding: 4px 6px;
             margin-bottom: 6px; background: #f3f3f9; border-left: 3px solid #557;
             color: #225; font-weight: 500; }
.loc { color: #888; font-size: 10px; margin-left: 8px; }
.opc { color: #058; }
.empty { color: #888; padding: 12px; }
.pass-hdr { font-weight: bold; color: #335; margin: 8px 0 2px 0; }
.func-hdr { color: #553; margin-left: 1em; font-size: 11px; }
.snap-inst { white-space: pre; margin-left: 2em; color: #111; }
.changed { background: #f0fff0; }
.tag { display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 10px; margin-left: 6px; vertical-align: 1px; }
.tag-final { background: #d6f5d6; color: #064; }
.tag-last  { background: #ffe0b3; color: #840; }
.tag-gone  { background: #f7c8c8; color: #800; }
table.idx { border-collapse: collapse; }
table.idx td, table.idx th { padding: 3px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 12px; }
"""

_SCRIPT_JS = """\
function renderHistory(loc) {
  var box = document.getElementById('history');
  var groups = (window.HIST || {})[loc];
  if (!groups || groups.length === 0) {
    box.innerHTML = '<div class="empty">No history for this location.</div>';
    return;
  }
  var parts = ['<div class="pass-hdr">Location: ' + escapeHtml(loc) + '</div>'];
  parts.push('<div class="empty">' + groups.length + ' distinct snapshot(s)</div>');
  for (var i = 0; i < groups.length; i++) {
    var g = groups[i];
    var tag = '';
    if (g.vanished) {
      tag = ' <span class="tag tag-gone">removed by/before this pass</span>';
    } else if (g.final) {
      tag = g.alive_at_end
            ? ' <span class="tag tag-final">final (alive at end)</span>'
            : ' <span class="tag tag-last">last seen here</span>';
    }
    parts.push('<div class="pass-hdr">seq=' + g.seq + ' ' + escapeHtml(g.pass)
               + tag + ' <span class="loc">on ' + escapeHtml(g.ir_unit) + '</span></div>');
    if (g.vanished) {
      parts.push('<div class="empty">(no instructions at this location)</div>');
      continue;
    }
    var lastBlock = '';
    for (var j = 0; j < g.insts.length; j++) {
      var it = g.insts[j];
      if (it.block !== lastBlock) {
        parts.push('<div class="func-hdr">block ' + escapeHtml(it.block) + ':</div>');
        lastBlock = it.block;
      }
      parts.push('<div class="snap-inst">' + escapeHtml(it.text) + '</div>');
    }
  }
  box.innerHTML = parts.join('');
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function(c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
function instsWithLoc(loc) {
  var all = document.querySelectorAll('.inst[data-loc]');
  var out = [];
  for (var i = 0; i < all.length; i++) {
    if (all[i].getAttribute('data-loc') === loc) out.push(all[i]);
  }
  return out;
}
function clearClass(cls) {
  var prev = document.getElementsByClassName(cls);
  while (prev.length) prev[0].classList.remove(cls);
}
function panelOf(el) {
  while (el && !(el.classList && el.classList.contains('panel')))
    el = el.parentElement;
  return el;
}
function alignPeerToClicked(clicked, peer) {
  var cp = panelOf(clicked), pp = panelOf(peer);
  if (!cp || !pp || cp === pp) return;
  // Vertical offset of the clicked line within its own panel's viewport.
  var anchor = clicked.getBoundingClientRect().top
             - cp.getBoundingClientRect().top;
  // Where the peer currently sits within its panel's viewport.
  var peerTop = peer.getBoundingClientRect().top
              - pp.getBoundingClientRect().top;
  // Scroll the peer's panel so the peer ends up at the same vertical
  // position as the clicked line. Clamped automatically by the browser.
  pp.scrollTop += peerTop - anchor;
}
function selectInst(el) {
  clearClass('selected');
  clearClass('linked');
  var loc = el.getAttribute('data-loc');
  if (!loc) return;
  var peers = instsWithLoc(loc);
  for (var i = 0; i < peers.length; i++) {
    peers[i].classList.add('selected');
    if (peers[i] !== el) alignPeerToClicked(el, peers[i]);
  }
  renderHistory(loc);
}
function hoverInst(el, enter) {
  var loc = el.getAttribute('data-loc');
  if (!loc) return;
  var peers = instsWithLoc(loc);
  for (var i = 0; i < peers.length; i++) {
    if (peers[i].classList.contains('selected')) continue;
    if (enter) peers[i].classList.add('linked');
    else peers[i].classList.remove('linked');
  }
}
function filterFuncs() {
  var q = document.getElementById('filter').value.toLowerCase();
  var items = document.querySelectorAll('#funcs li');
  for (var i = 0; i < items.length; i++) {
    var t = items[i].textContent.toLowerCase();
    items[i].style.display = (q === '' || t.indexOf(q) >= 0) ? '' : 'none';
  }
}
window.addEventListener('DOMContentLoaded', function() {
  var first = document.querySelector('#initial .inst[data-loc]');
  if (first) selectInst(first);
});
"""


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or "x"


def _esc(s: str) -> str:
    return html.escape(s, quote=False)


def _functions(con: sqlite3.Connection) -> List[Tuple[str, str, int, int, str]]:
    """Return list of ``(function, file_path, n_insts, n_passes, page)``."""
    rows = con.execute(
        f"SELECT i.function AS function, f.path AS file_path, "
        f"COUNT(*) AS n_insts, COUNT(DISTINCT i.pass_id) AS n_passes "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function != '' AND p.kind = 'ir' "
        f"GROUP BY i.function, f.id "
        f"ORDER BY f.path, i.function"
    ).fetchall()
    out = []
    used: Dict[str, int] = {}
    for r in rows:
        base = "fn-" + _safe_filename(r["function"])
        n = used.get(base, 0)
        used[base] = n + 1
        page = base + (f"-{n}" if n else "") + ".html"
        out.append(
            (r["function"], r["file_path"], int(r["n_insts"]), int(r["n_passes"]), page)
        )
    return out


def _initial_seq(con: sqlite3.Connection, function: str) -> int:
    """Smallest recorded pass ``seq`` for a function whose phase is the
    initial capture. We intentionally exclude ``phase='final'`` rows so that
    a function first seen in the synthetic final snapshot (e.g. one that had
    no activity during the pipeline) does not get treated as an "initial"."""
    row = con.execute(
        f"SELECT MIN(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind = 'ir' AND p.phase <> 'final'",
        (function,),
    ).fetchone()
    return -1 if row is None or row["s"] is None else int(row["s"])


def _final_seq(con: sqlite3.Connection, function: str) -> int:
    """Return the ``seq`` of the synthetic ``phase='final'`` pass that has
    rows for this function, or ``-1`` when the DB predates that recorder
    change or the function did not survive to the final snapshot."""
    row = con.execute(
        f"SELECT MAX(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind = 'ir' AND p.phase = 'final'",
        (function,),
    ).fetchone()
    return -1 if row is None or row["s"] is None else int(row["s"])


def _final_ir_rows(
    con: sqlite3.Connection, function: str, seq: int
) -> List[sqlite3.Row]:
    return con.execute(
        f"SELECT i.basicblock, i.inst_seq, i.opcode, i.inst_text, "
        f"f.path AS file_path, i.line, i.col "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function = ? AND p.kind = 'ir' AND p.seq = ? "
        f"ORDER BY i.id",
        (function, seq),
    ).fetchall()


def _initial_ir(con: sqlite3.Connection, function: str, seq: int) -> List[sqlite3.Row]:
    # Order by recording order (``i.id``) rather than ``(basicblock, inst_seq)``
    # because the cost-improvement recorder often labels every block as
    # ``<unnamed>`` and ``inst_seq`` is the *per-block* instruction index that
    # restarts at 0 for each new block. Sorting on those columns therefore
    # interleaves instructions from different blocks. Insertion order on a
    # single ``writeInstructionsInFunction`` walk preserves the natural
    # block-by-block order.
    return con.execute(
        f"SELECT i.basicblock, i.inst_seq, i.opcode, i.inst_text, "
        f"f.path AS file_path, i.line, i.col "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function = ? AND p.kind = 'ir' AND p.seq = ? "
        f"ORDER BY i.id",
        (function, seq),
    ).fetchall()


def _history(
    con: sqlite3.Connection, function: str
) -> Dict[str, List[Dict[str, object]]]:
    rows = con.execute(
        f"SELECT f.path, i.line, i.col, p.seq, p.pass_class, p.ir_unit, "
        f"i.basicblock, i.inst_seq, i.inst_text "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function = ? AND p.kind = 'ir' "
        f"ORDER BY i.line, i.col, p.seq, i.basicblock, i.inst_seq",
        (function,),
    ).fetchall()

    # Maximum seq for which this function has any recorded rows, used to
    # decide whether a location reached the end of the pipeline or vanished.
    func_max_row = con.execute(
        f"SELECT MAX(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind = 'ir'",
        (function,),
    ).fetchone()
    func_max_seq = (
        -1
        if func_max_row is None or func_max_row["s"] is None
        else int(func_max_row["s"])
    )

    # Group rows by location key, then by seq within each location.
    by_loc: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    pass_meta: Dict[int, Tuple[str, str]] = {}
    for r in rows:
        key = f"{r['path']}|{int(r['line'])}|{int(r['col'])}"
        by_loc.setdefault(key, {}).setdefault(int(r["seq"]), []).append(
            {"block": r["basicblock"] or "", "text": r["inst_text"] or ""}
        )
        pass_meta[int(r["seq"])] = (r["pass_class"] or "", r["ir_unit"] or "")

    out: Dict[str, List[Dict[str, object]]] = {}
    for key, by_seq in by_loc.items():
        seqs = sorted(by_seq)
        loc_max_seq = seqs[-1]

        groups: List[Dict[str, object]] = []
        last_fp = None
        for seq in seqs:
            insts = by_seq[seq]
            fp = "\n".join(it["text"] for it in insts)
            is_last = seq == loc_max_seq
            if fp == last_fp and not is_last:
                continue
            pass_name, ir_unit = pass_meta[seq]
            entry: Dict[str, object] = {
                "seq": seq,
                "pass": pass_name,
                "ir_unit": ir_unit,
                "insts": insts,
            }
            if is_last:
                # Tag the final recorded snapshot so the UI can highlight
                # whether the location survived to the end of the pipeline
                # or was dropped by a later pass.
                entry["final"] = True
                entry["alive_at_end"] = loc_max_seq == func_max_seq
            groups.append(entry)
            last_fp = fp

        # If the location vanished before the function's last pass, append a
        # synthetic record naming the first pass after it that was recorded
        # (best estimate of the pass that removed it).
        if loc_max_seq < func_max_seq:
            after = None
            for s in sorted(pass_meta):
                if s > loc_max_seq:
                    after = s
                    break
            if after is not None:
                pass_name, ir_unit = pass_meta[after]
                groups.append(
                    {
                        "seq": after,
                        "pass": pass_name,
                        "ir_unit": ir_unit,
                        "insts": [],
                        "vanished": True,
                    }
                )

        out[key] = groups
    return out


def _render_func_list(
    funcs: Sequence[Tuple[str, str, int, int, str]], current: str
) -> str:
    parts: List[str] = []
    parts.append(
        '<input id="filter" type="text" placeholder="Filter functions..." '
        'oninput="filterFuncs()">'
    )
    cur_file = None
    parts.append("<ul>")
    for fn, fpath, n_i, n_p, page in funcs:
        if fpath != cur_file:
            if cur_file is not None:
                parts.append("</ul>")
            parts.append(
                '<div class="file">' + _esc(os.path.basename(fpath) or fpath) + "</div>"
            )
            parts.append("<ul>")
            cur_file = fpath
        klass = ' class="current"' if fn == current else ""
        parts.append(
            f'<li><a href="{page}"{klass}>{_esc(fn)}</a> '
            f'<span class="loc">{n_i}r {n_p}p</span></li>'
        )
    parts.append("</ul>")
    return "".join(parts)


def _render_initial_ir(
    rows: Sequence[sqlite3.Row],
    seq: int,
    color_locs: Optional[FrozenSet[str]] = None,
) -> str:
    return _render_ir_panel(rows, f"initial snapshot: seq={seq}", color_locs)


_DEFINE_RE = re.compile(r"^\s*define\b[^@]*@([A-Za-z0-9_.$]+)\s*\(")
_DILOC_RE = re.compile(
    r"^!(\d+)\s*=\s*!DILocation\(\s*line:\s*(\d+)(?:,\s*column:\s*(\d+))?"
)
_DBG_REF_RE = re.compile(r"!dbg !(\d+)")


def _row_loc(r: sqlite3.Row) -> Optional[str]:
    line = int(r["line"])
    if line <= 0:
        return None
    return f'{r["file_path"]}|{line}|{int(r["col"])}'


def _render_ir_panel(
    rows: Sequence[sqlite3.Row],
    header: str,
    color_locs: Optional[FrozenSet[str]] = None,
) -> str:
    """Shared renderer for a single-pass IR snapshot (used by both the
    Initial and Final panels). Groups into pseudo-blocks by ``inst_seq``
    resetting to zero, since the compact printer often emits ``<unnamed>``
    instead of a real block name."""
    if not rows:
        return '<div class="empty">No IR recorded.</div>'
    parts: List[str] = [f'<div class="loc">{_esc(header)}</div>']

    # Pull the synthetic signature row (basicblock=='<sig>') to the top so
    # the panel opens with the textual-IR-style ``define ... @name(...)``.
    sig_row = next((r for r in rows if (r["basicblock"] or "") == "<sig>"), None)
    body_rows = [r for r in rows if (r["basicblock"] or "") != "<sig>"]
    if sig_row is not None:
        parts.append(
            '<div class="signature">' f'{_esc(sig_row["inst_text"] or "")}' "</div>"
        )

    bb_index = 0
    last_seq = -1
    last_bb_name: Optional[str] = None
    for idx, r in enumerate(body_rows):
        bb_name = r["basicblock"] or ""
        is_real_name = bool(bb_name) and bb_name != "<unnamed>"
        seq_resets = int(r["inst_seq"]) == 0 and last_seq >= 0
        starts_new_block = (
            idx == 0 or seq_resets or (is_real_name and bb_name != last_bb_name)
        )
        if starts_new_block:
            label = bb_name if is_real_name else f"bb{bb_index}"
            parts.append('<div class="bb">' + _esc(label) + ":</div>")
            bb_index += 1
            last_bb_name = bb_name
        last_seq = int(r["inst_seq"])

        loc = f'{r["file_path"]}|{int(r["line"])}|{int(r["col"])}'
        has_loc = int(r["line"]) > 0
        klass = "inst" + ("" if has_loc else " no-loc")
        attrs = (
            f' data-loc="{_esc(loc)}" onclick="selectInst(this)"'
            f' onmouseenter="hoverInst(this,true)"'
            f' onmouseleave="hoverInst(this,false)"'
            if has_loc
            else ""
        )
        # Inline a stable per-loc background only for locations that show
        # up in both the Initial and Final panels — that is the cross-panel
        # mapping the user wants to see at a glance. Lone-side locations
        # stay the default color so the noise stays manageable.
        style = ""
        if has_loc and color_locs is not None and loc in color_locs:
            style = f' style="background:{_loc_color(loc)}"'
        loc_str = (
            f'<span class="loc">{int(r["line"])}:{int(r["col"])}</span>'
            if has_loc
            else '<span class="loc">no loc</span>'
        )
        parts.append(
            f'<div class="{klass}"{attrs}{style}>'
            f'<span class="opc">{_esc(r["opcode"] or "")}</span>  '
            f'{_esc(r["inst_text"] or "")}{loc_str}</div>'
        )
    return "".join(parts)


def parse_ll_functions(path: str) -> Tuple[Dict[str, str], Dict[int, Tuple[int, int]]]:
    """Return ``({function_name: body_text}, {metadata_id: (line, col)})`` from
    an LLVM textual IR file. Debug-location mapping only includes direct
    ``!DILocation`` nodes (not ``DILexicalBlock``-scoped chains); that is
    enough for ``--add-ir-tracker-locs``-produced IR which emits one
    ``!DILocation`` per synthetic line."""
    funcs: Dict[str, str] = {}
    dilocs: Dict[int, Tuple[int, int]] = {}
    cur: Optional[str] = None
    buf: List[str] = []
    depth = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if cur is None:
                m = _DILOC_RE.match(line)
                if m:
                    line_n = int(m.group(2))
                    col_n = int(m.group(3)) if m.group(3) is not None else 0
                    dilocs[int(m.group(1))] = (line_n, col_n)
                    continue
                dm = _DEFINE_RE.match(line)
                if not dm:
                    continue
                cur = dm.group(1)
                buf = [line.rstrip("\n")]
                depth = line.count("{") - line.count("}")
                if depth <= 0:
                    funcs[cur] = "\n".join(buf)
                    cur = None
                    buf = []
                continue
            buf.append(line.rstrip("\n"))
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                funcs[cur] = "\n".join(buf)
                cur = None
                buf = []
    return funcs, dilocs


def _render_final_ir(
    rows: Sequence[sqlite3.Row],
    seq: int,
    color_locs: Optional[FrozenSet[str]] = None,
) -> str:
    if not rows or seq < 0:
        return (
            '<div class="empty">No final snapshot for this function. '
            "Ensure the tracker DB was produced by a recorder that emits a "
            "<code>phase='final'</code> pass at teardown.</div>"
        )
    return _render_ir_panel(rows, f"final snapshot: seq={seq}", color_locs)


def _render_function_page(
    function: str,
    file_path: str,
    funcs: Sequence[Tuple[str, str, int, int, str]],
    initial_rows: Sequence[sqlite3.Row],
    initial_seq: int,
    final_rows: Sequence[sqlite3.Row],
    final_seq: int,
    history: Dict[str, List[Dict[str, object]]],
) -> str:
    # Locations shared by Initial and Final get a stable per-loc background
    # color so the cross-panel mapping is visible at a glance.
    initial_locs = {l for l in (_row_loc(r) for r in initial_rows) if l}
    final_locs = {l for l in (_row_loc(r) for r in final_rows) if l}
    shared_locs: FrozenSet[str] = frozenset(initial_locs & final_locs)
    func_list = _render_func_list(funcs, function)
    initial = _render_initial_ir(initial_rows, initial_seq, shared_locs)
    final = _render_final_ir(final_rows, final_seq, shared_locs)
    hist_json = json.dumps(history, separators=(",", ":"))
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(function)}</title>"
        "<link rel='stylesheet' href='style.css'>"
        f"<script>{_SCRIPT_JS}</script>"
        "</head><body>"
        f"<header><a href='index.html'>&larr; index</a> &nbsp; "
        f"<b>{_esc(function)}</b> "
        f"<span class='loc'>{_esc(file_path)}</span></header>"
        "<div class='layout'>"
        f"<div id='funcs' class='panel'>{func_list}</div>"
        f"<div id='initial' class='panel'><h3>Initial IR</h3>{initial}</div>"
        f"<div id='final' class='panel'><h3>Final IR</h3>{final}</div>"
        "<div id='history' class='panel'>"
        "<h3>Pass history</h3>"
        "<div class='empty'>Click an instruction in either IR panel to "
        "see its pass history.</div>"
        "</div>"
        "</div>"
        f"<script>window.HIST={hist_json};</script>"
        "</body></html>"
    )


def _render_index(
    funcs: Sequence[Tuple[str, str, int, int, str]],
    pass_count: int,
    inst_count: int,
) -> str:
    rows = []
    cur_file = None
    for fn, fpath, n_i, n_p, page in funcs:
        if fpath != cur_file:
            cur_file = fpath
            rows.append(
                f"<tr><th colspan='4'>{_esc(fpath)}</th></tr>"
                "<tr><th>Function</th><th>Inst rows</th>"
                "<th>Passes touched</th><th></th></tr>"
            )
        rows.append(
            f"<tr><td><a href='{page}'>{_esc(fn)}</a></td>"
            f"<td>{n_i}</td><td>{n_p}</td><td></td></tr>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ir-tracker report</title>"
        "<link rel='stylesheet' href='style.css'></head><body>"
        "<header>ir-tracker report &mdash; "
        f"{pass_count} pass snapshots, {inst_count} instruction rows, "
        f"{len(funcs)} functions</header>"
        "<div style='padding:10px'>"
        f"<table class='idx'>{''.join(rows)}</table>"
        "</div></body></html>"
    )


def generate_html(
    con: sqlite3.Connection,
    output_dir: str,
    source_dirs: Sequence[str],
    all_passes: bool,
    no_highlight: bool,
    file_filter: str = "",
) -> int:
    # ``source_dirs``, ``all_passes`` and ``no_highlight`` are accepted for CLI
    # compatibility but unused in the function-centric layout (no source view,
    # always emits per-location dedup, no syntax highlighting).
    del source_dirs, all_passes, no_highlight

    if irtrackdb.get_schema_version(con) < 1:
        print("ir-tracker: unsupported schema version", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)

    funcs = _functions(con)
    if file_filter:
        needle = file_filter.lower()
        funcs = [t for t in funcs if needle in t[1].lower()]
    if not funcs:
        print("ir-tracker: no functions with instructions in DB", file=sys.stderr)
        return 1

    pass_count = int(
        con.execute(
            f"SELECT COUNT(*) AS c FROM {irtrackdb.T_PASSES} WHERE kind = 'ir'"
        ).fetchone()["c"]
    )
    inst_count = int(
        con.execute(
            f"SELECT COUNT(*) AS c FROM {irtrackdb.T_INSTR} i "
            f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
            f"WHERE p.kind = 'ir'"
        ).fetchone()["c"]
    )

    with open(os.path.join(output_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(_STYLE_CSS)

    for function, file_path, _ni, _np, page in funcs:
        iseq = _initial_seq(con, function)
        initial = _initial_ir(con, function, iseq) if iseq >= 0 else []
        fseq = _final_seq(con, function)
        final = _final_ir_rows(con, function, fseq) if fseq >= 0 else []
        hist = _history(con, function)
        html_text = _render_function_page(
            function,
            file_path,
            funcs,
            initial,
            iseq,
            final,
            fseq,
            hist,
        )
        with open(os.path.join(output_dir, page), "w", encoding="utf-8") as f:
            f.write(html_text)

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(_render_index(funcs, pass_count, inst_count))

    print(f"ir-tracker: wrote {len(funcs)} function page(s) + index to {output_dir}")
    return 0
