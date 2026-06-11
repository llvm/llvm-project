# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HTML report generator for ir-tracker SQLite databases.

Generates a static, multi-panel site organized by function:

  * ``index.html`` — function list grouped by file, with row counts.
  * One ``<safe>.html`` per function containing:

      - Function list, source, initial IR, final MIR, and history panels.
        Each panel is collapsible, and each instruction is clickable.
      - History panel: when an instruction is clicked, the panel shows the
        full pass-by-pass IR/MIR history of all instructions sharing that
        source location, deduplicated when text is unchanged.
      - Optional ``assembly.html`` page for final assembly output.

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
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

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
.panel { overflow: auto; padding: 8px 12px; min-width: 0; }
.layout.resizing, .layout.resizing * { cursor: col-resize !important; user-select: none; }
.panel.collapsed { width: 34px !important; flex: 0 0 34px; min-width: 34px; padding: 8px 4px; overflow: hidden; }
.panel.collapsed .panel-body { display: none; }
.panel.collapsed .panel-title { writing-mode: vertical-rl; text-orientation: mixed; white-space: nowrap; }
.panel.collapsed .panel-header { display: block; }
.panel-header { display: flex; align-items: center; gap: 6px; margin: 0 0 6px 0; }
.panel-title { flex: 1; font-size: 11px; text-transform: uppercase; color: #666; letter-spacing: .5px; }
.panel-toggle { border: 1px solid #bbb; background: #f8f8f8; color: #555; border-radius: 3px; width: 18px; height: 18px; line-height: 14px; padding: 0; cursor: pointer; }
.splitter { flex: 0 0 5px; width: 5px; cursor: col-resize; background: #e5e5ea; border-left: 1px solid #d0d0d8; border-right: 1px solid #d0d0d8; }
.splitter:hover, .splitter.active { background: #c8d6f0; }
#funcs { width: 14%; border-right: 1px solid #ccc; background: #f7f7fb; font-size: 12px; }
#funcs ul { list-style: none; padding-left: 6px; margin: 4px 0 12px 0; }
#funcs li { margin: 1px 0; }
#funcs a { text-decoration: none; color: #224; }
#funcs a.current { font-weight: bold; color: #b00; }
#funcs .file { font-weight: bold; margin-top: 6px; color: #555; }
#filter { width: 95%; padding: 3px; margin-top: 4px; }
#source { width: 18%; border-right: 1px solid #ccc; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
#initial, #final { width: 22%; border-right: 1px solid #ccc; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
#history { width: 24%; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.source-file { color: #666; font-size: 11px; margin-bottom: 6px; word-break: break-all; }
.source-code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; }
.src-line { display: flex; white-space: pre; min-height: 16px; border-radius: 3px; cursor: pointer; }
.src-line:hover { background: #eef; }
.src-line.src-selected { background: #ffe080; outline: 1px solid #caa040; }
.src-num { flex: 0 0 3.2em; color: #999; text-align: right; padding-right: 8px; user-select: none; }
.src-text { flex: 1; }
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
.kind { display: inline-block; min-width: 2.6em; color: #666; font-size: 10px; text-transform: uppercase; }
.tag { display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 10px; margin-left: 6px; vertical-align: 1px; }
.tag-final { background: #d6f5d6; color: #064; }
.tag-last  { background: #ffe0b3; color: #840; }
.tag-gone  { background: #f7c8c8; color: #800; }
table.idx { border-collapse: collapse; }
table.idx td, table.idx th { padding: 3px 8px; border-bottom: 1px solid #eee; text-align: left; font-size: 12px; }
pre.asm { margin: 0; padding: 10px; font: 12px ui-monospace, Menlo, Consolas, monospace; white-space: pre; }
"""

_SCRIPT_JS = """\
function renderHistory(loc) {
  var box = document.getElementById('history-body');
  var groups = (window.HIST || {})[loc];
  if (!groups || groups.length === 0) {
    box.innerHTML = '<div class="empty">No history for this location.</div>';
    return;
  }
  var parts = [];
  appendHistory(parts, loc, groups);
  box.innerHTML = parts.join('');
}
function appendHistory(parts, loc, groups) {
  parts.push('<div class="pass-hdr">Location: ' + escapeHtml(loc) + '</div>');
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
    parts.push('<div class="pass-hdr">seq=' + g.seq + ' <span class="kind">'
               + escapeHtml(g.kind || 'ir') + '</span> ' + escapeHtml(g.pass)
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
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, function(c) {
    return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c];
  });
}
function locLine(loc) {
  var parts = String(loc).split('|');
  return parts.length >= 3 ? parts[parts.length - 2] : '';
}
function instsWithLoc(loc) {
  var all = document.querySelectorAll('.inst[data-loc]');
  var out = [];
  for (var i = 0; i < all.length; i++) {
    if (all[i].getAttribute('data-loc') === loc) out.push(all[i]);
  }
  return out;
}
function instsWithLine(line) {
  var all = document.querySelectorAll('.inst[data-loc]');
  var out = [];
  line = String(line);
  for (var i = 0; i < all.length; i++) {
    if (locLine(all[i].getAttribute('data-loc')) === line) out.push(all[i]);
  }
  return out;
}
function locsForLine(line) {
  var insts = instsWithLine(line);
  var seen = {};
  var out = [];
  for (var i = 0; i < insts.length; i++) {
    var loc = insts[i].getAttribute('data-loc');
    if (!seen[loc]) {
      seen[loc] = true;
      out.push(loc);
    }
  }
  return out;
}
function sourceLinesWithLine(line) {
  return document.querySelectorAll('.src-line[data-line="' + String(line) + '"]');
}
function clearClass(cls) {
  var prev = document.getElementsByClassName(cls);
  while (prev.length) prev[0].classList.remove(cls);
}
function togglePanel(id) {
  var panel = document.getElementById(id);
  if (!panel) return;
  panel.classList.toggle('collapsed');
  var btn = panel.querySelector('.panel-toggle');
  if (btn) btn.textContent = panel.classList.contains('collapsed') ? '+' : '-';
  fillRemainingPanel();
}
function panelWidth(panel) {
  return panel.getBoundingClientRect().width;
}
function setPanelWidth(panel, width) {
  panel.style.width = width + 'px';
  panel.style.flex = '0 0 ' + width + 'px';
}
function fillRemainingPanel() {
  var panels = document.querySelectorAll('.layout > .panel');
  var fill = null;
  for (var i = 0; i < panels.length; i++) {
    if (!panels[i].classList.contains('collapsed')) fill = panels[i];
  }
  for (var j = 0; j < panels.length; j++) {
    if (panels[j].classList.contains('collapsed')) continue;
    panels[j].style.flexGrow = panels[j] === fill ? '1' : '0';
  }
}
function initSplitters() {
  var splitters = document.querySelectorAll('.splitter');
  for (var i = 0; i < splitters.length; i++) {
    splitters[i].addEventListener('pointerdown', function(e) {
      var left = document.getElementById(this.getAttribute('data-left'));
      var right = document.getElementById(this.getAttribute('data-right'));
      if (!left || !right || left.classList.contains('collapsed') ||
          right.classList.contains('collapsed'))
        return;
      e.preventDefault();
      var splitter = this;
      var layout = splitter.parentElement;
      var startX = e.clientX;
      var startLeft = panelWidth(left);
      var startRight = panelWidth(right);
      var min = 80;
      splitter.classList.add('active');
      if (layout) layout.classList.add('resizing');
      splitter.setPointerCapture(e.pointerId);

      function move(ev) {
        var dx = ev.clientX - startX;
        var nextLeft = Math.max(min, startLeft + dx);
        var nextRight = Math.max(min, startRight - dx);
        var used = nextLeft + nextRight;
        var total = startLeft + startRight;
        if (used > total) {
          if (nextLeft === min) nextRight = total - min;
          else nextLeft = total - min;
        }
        setPanelWidth(left, nextLeft);
        setPanelWidth(right, nextRight);
      }
      function up(ev) {
        splitter.releasePointerCapture(ev.pointerId);
        splitter.classList.remove('active');
        if (layout) layout.classList.remove('resizing');
        fillRemainingPanel();
        splitter.removeEventListener('pointermove', move);
        splitter.removeEventListener('pointerup', up);
        splitter.removeEventListener('pointercancel', up);
      }
      splitter.addEventListener('pointermove', move);
      splitter.addEventListener('pointerup', up);
      splitter.addEventListener('pointercancel', up);
    });
  }
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
  clearClass('src-selected');
  var loc = el.getAttribute('data-loc');
  if (!loc) return;
  var peers = instsWithLoc(loc);
  for (var i = 0; i < peers.length; i++) {
    peers[i].classList.add('selected');
    if (peers[i] !== el) alignPeerToClicked(el, peers[i]);
  }
  highlightSource(loc);
  renderHistory(loc);
}
function highlightSource(loc) {
  var line = locLine(loc);
  if (!line) return;
  var sourcePanel = document.getElementById('source');
  var lines = sourceLinesWithLine(line);
  for (var i = 0; i < lines.length; i++) {
    lines[i].classList.add('src-selected');
    if (sourcePanel && !sourcePanel.classList.contains('collapsed'))
      alignPeerToClicked(document.querySelector('.inst.selected') || lines[i], lines[i]);
  }
}
function renderLineHistory(line) {
  var box = document.getElementById('history-body');
  var locs = locsForLine(line);
  if (locs.length === 0) {
    box.innerHTML = '<div class="empty">No history for this source line.</div>';
    return;
  }
  var parts = ['<div class="pass-hdr">Source line: ' + escapeHtml(line) + '</div>'];
  parts.push('<div class="empty">' + locs.length + ' source location(s)</div>');
  for (var i = 0; i < locs.length; i++) {
    var groups = (window.HIST || {})[locs[i]] || [];
    if (groups.length === 0) continue;
    appendHistory(parts, locs[i], groups);
  }
  box.innerHTML = parts.join('');
}
function selectSourceLine(el) {
  var line = el.getAttribute('data-line');
  var page = el.getAttribute('data-page');
  var current = window.CURRENT_PAGE || '';
  var peers = instsWithLine(line);
  if (peers.length === 0 && page && page !== current) {
    window.location.href = page + '#line-' + encodeURIComponent(line);
    return;
  }

  clearClass('selected');
  clearClass('linked');
  clearClass('src-selected');
  var sourceLines = sourceLinesWithLine(line);
  for (var i = 0; i < sourceLines.length; i++)
    sourceLines[i].classList.add('src-selected');

  var alignedPanels = [];
  for (var j = 0; j < peers.length; j++) {
    peers[j].classList.add('selected');
    var p = panelOf(peers[j]);
    if (p && alignedPanels.indexOf(p) < 0) {
      alignPeerToClicked(el, peers[j]);
      alignedPanels.push(p);
    }
  }
  renderLineHistory(line);
}
function hoverSourceLine(el, enter) {
  var peers = instsWithLine(el.getAttribute('data-line'));
  for (var i = 0; i < peers.length; i++) {
    if (peers[i].classList.contains('selected')) continue;
    if (enter) peers[i].classList.add('linked');
    else peers[i].classList.remove('linked');
  }
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
  initSplitters();
  fillRemainingPanel();
  if (window.location.hash.indexOf('#line-') === 0) {
    var line = decodeURIComponent(window.location.hash.slice(6));
    var sourceLine = document.querySelector('.src-line[data-line="' + line + '"]');
    if (sourceLine) {
      selectSourceLine(sourceLine);
      return;
    }
  }
  var first = document.querySelector('#initial .inst[data-loc]');
  if (first) selectInst(first);
});
"""


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or "x"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _render_panel(panel_id: str, title: str, body: str) -> str:
    return (
        f'<div id="{_esc(panel_id)}" class="panel">'
        '<div class="panel-header">'
        f'<button class="panel-toggle" onclick="togglePanel(\'{_esc(panel_id)}\')" '
        f'title="Toggle {_esc(title)}">-</button>'
        f'<div class="panel-title">{_esc(title)}</div>'
        "</div>"
        f'<div id="{_esc(panel_id)}-body" class="panel-body">{body}</div>'
        "</div>"
    )


def _render_splitter(left: str, right: str) -> str:
    return (
        f'<div class="splitter" data-left="{_esc(left)}" '
        f'data-right="{_esc(right)}" title="Resize panels"></div>'
    )


def _resolve_source_path(file_path: str, source_dirs: Sequence[str]) -> str:
    if file_path and os.path.isfile(file_path):
        return file_path
    base = os.path.basename(file_path)
    for source_dir in source_dirs:
        candidate = os.path.join(source_dir, base)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _render_source_panel(
    file_path: str, source_dirs: Sequence[str], line_pages: Dict[int, str]
) -> str:
    source_path = _resolve_source_path(file_path, source_dirs)
    if not source_path:
        return '<div class="empty">Source file not found.</div>'

    parts: List[str] = [
        f'<div class="source-file">{_esc(source_path)}</div>',
        '<div class="source-code">',
    ]
    with open(source_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            data_page = (
                f' data-page="{_esc(line_pages[line_no])}"'
                if line_no in line_pages
                else ""
            )
            parts.append(
                f'<div class="src-line" data-line="{line_no}"{data_page} '
                'onclick="selectSourceLine(this)" '
                'onmouseenter="hoverSourceLine(this,true)" '
                'onmouseleave="hoverSourceLine(this,false)">'
                f'<span class="src-num">{line_no}</span>'
                f'<span class="src-text">{_esc(line.rstrip(chr(10)))}</span>'
                "</div>"
            )
    parts.append("</div>")
    return "".join(parts)


def _functions(con: sqlite3.Connection) -> List[Tuple[str, str, int, int, str]]:
    """Return list of ``(function, file_path, n_insts, n_passes, page)``."""
    rows = con.execute(
        f"SELECT i.function AS function, f.path AS file_path, "
        f"COUNT(*) AS n_insts, COUNT(DISTINCT i.pass_id) AS n_passes "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function != '' AND p.kind IN ('ir', 'mir') "
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


def _source_line_pages(
    con: sqlite3.Connection, page_by_function: Dict[str, str]
) -> Dict[Tuple[str, int], str]:
    """Return the best target function page for each tracked source line."""
    rows = con.execute(
        f"SELECT f.path AS file_path, i.line, i.function, COUNT(*) AS n "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function != '' AND i.line > 0 AND p.kind IN ('ir', 'mir') "
        f"GROUP BY f.path, i.line, i.function "
        f"ORDER BY f.path, i.line, n DESC, i.function"
    ).fetchall()
    out: Dict[Tuple[str, int], str] = {}
    seen: Set[Tuple[str, int]] = set()
    for r in rows:
        key = (r["file_path"], int(r["line"]))
        if key in seen:
            continue
        page = page_by_function.get(r["function"])
        if page:
            out[key] = page
            seen.add(key)
    return out


def _initial_seq(con: sqlite3.Connection, function: str, kind: str) -> int:
    """Smallest recorded pass ``seq`` for a function whose phase is the
    initial capture. We intentionally exclude ``phase='final'`` rows so that
    a function first seen in the synthetic final snapshot (e.g. one that had
    no activity during the pipeline) does not get treated as an "initial"."""
    row = con.execute(
        f"SELECT MIN(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind = ? AND p.phase <> 'final'",
        (function, kind),
    ).fetchone()
    return -1 if row is None or row["s"] is None else int(row["s"])


def _last_seq(con: sqlite3.Connection, function: str, kind: str) -> int:
    row = con.execute(
        f"SELECT MAX(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind = ?",
        (function, kind),
    ).fetchone()
    return -1 if row is None or row["s"] is None else int(row["s"])


def _snapshot_rows(
    con: sqlite3.Connection, function: str, kind: str, seq: int
) -> List[sqlite3.Row]:
    # Order by recording order (``i.id``) rather than ``(basicblock, inst_seq)``
    # because the compact recorder can label every block as ``<unnamed>`` and
    # ``inst_seq`` restarts at 0 for each new block.
    return con.execute(
        f"SELECT i.basicblock, i.inst_seq, i.opcode, i.inst_text, "
        f"f.path AS file_path, i.line, i.col "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function = ? AND p.kind = ? AND p.seq = ? "
        f"ORDER BY i.id",
        (function, kind, seq),
    ).fetchall()


def _history(
    con: sqlite3.Connection, function: str
) -> Dict[str, List[Dict[str, object]]]:
    rows = con.execute(
        f"SELECT f.path, i.line, i.col, p.seq, p.kind, p.pass_class, p.ir_unit, "
        f"i.basicblock, i.inst_seq, i.inst_text "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"JOIN {irtrackdb.T_FILES} f ON i.file_id = f.id "
        f"WHERE i.function = ? AND p.kind IN ('ir', 'mir') "
        f"ORDER BY i.line, i.col, p.seq, i.basicblock, i.inst_seq",
        (function,),
    ).fetchall()

    # Maximum seq for which this function has any recorded rows, used to
    # decide whether a location reached the end of the pipeline or vanished.
    func_max_row = con.execute(
        f"SELECT MAX(p.seq) AS s "
        f"FROM {irtrackdb.T_INSTR} i "
        f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
        f"WHERE i.function = ? AND p.kind IN ('ir', 'mir')",
        (function,),
    ).fetchone()
    func_max_seq = (
        -1
        if func_max_row is None or func_max_row["s"] is None
        else int(func_max_row["s"])
    )

    # Group rows by location key, then by seq within each location.
    by_loc: Dict[str, Dict[int, List[Dict[str, str]]]] = {}
    pass_meta: Dict[int, Tuple[str, str, str]] = {}
    for r in rows:
        key = f"{r['path']}|{int(r['line'])}|{int(r['col'])}"
        by_loc.setdefault(key, {}).setdefault(int(r["seq"]), []).append(
            {"block": r["basicblock"] or "", "text": r["inst_text"] or ""}
        )
        pass_meta[int(r["seq"])] = (
            r["kind"] or "",
            r["pass_class"] or "",
            r["ir_unit"] or "",
        )

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
            kind, pass_name, ir_unit = pass_meta[seq]
            entry: Dict[str, object] = {
                "seq": seq,
                "kind": kind,
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
                kind, pass_name, ir_unit = pass_meta[after]
                groups.append(
                    {
                        "seq": after,
                        "kind": kind,
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


_DIFILE_RE = re.compile(
    r'^!(\d+)\s*=\s*!DIFile\(filename:\s*"([^"]+)",\s*directory:\s*"([^"]*)"'
)
_DILOC_RE = re.compile(
    r"^!(\d+)\s*=\s*!DILocation\(\s*line:\s*(\d+)(?:,\s*column:\s*(\d+))?"
)
_DILOC_SCOPE_RE = re.compile(
    r"^!(\d+)\s*=\s*!DILocation\(\s*line:\s*(\d+)"
    r"(?:,\s*column:\s*(\d+))?.*?\bscope:\s*!(\d+)"
)
_DIMETA_FILE_REF_RE = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?![^(]+\(.*\bfile:\s*!(\d+)"
)
_DIMETA_SCOPE_REF_RE = re.compile(
    r"^!(\d+)\s*=\s*(?:distinct\s+)?![^(]+\(.*\bscope:\s*!(\d+)"
)
_MIR_DBG_REF_RE = re.compile(r",?\s*debug-location\s+!(\d+)")


def _row_loc(r: sqlite3.Row) -> Optional[str]:
    line = int(r["line"])
    if line <= 0:
        return None
    return f'{r["file_path"]}|{line}|{int(r["col"])}'


def _render_inst_panel(
    rows: Sequence[sqlite3.Row],
    header: str,
    color_locs: Optional[FrozenSet[str]] = None,
) -> str:
    """Shared renderer for a single-pass IR or MIR snapshot. Groups into
    pseudo-blocks by ``inst_seq``
    resetting to zero, since the compact printer often emits ``<unnamed>``
    instead of a real block name."""
    if not rows:
        return '<div class="empty">No instructions recorded.</div>'
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


def _mir_opcode(inst_text: str) -> str:
    head = inst_text.split("=", 1)[1] if "=" in inst_text else inst_text
    flags = {
        "contract",
        "dead",
        "exact",
        "nofpexcept",
        "killed",
        "nsw",
        "nuw",
        "renamable",
        "implicit-def",
        "implicit",
        "undef",
    }
    for part in head.strip().split():
        token = part.rstrip(",")
        if token not in flags:
            return token
    return ""


def parse_mir_snapshots(path: str) -> Dict[str, List[Dict[str, object]]]:
    """Return final MIR rows by function from an LLVM MIR file.

    The parser is intentionally small and handles the textual MIR that llc
    writes for ``-stop-after=...``. It keeps instruction text close to the MIR
    file while stripping ``debug-location`` suffixes into file/line/column rows
    compatible with the DB-backed renderer.
    """
    default_file = "<synthetic>"
    files_by_id: Dict[int, str] = {}
    scope_file_refs: Dict[int, int] = {}
    scope_parent_refs: Dict[int, int] = {}
    dilocs: Dict[int, Tuple[int, int, Optional[int]]] = {}
    rows_by_func: Dict[str, List[Dict[str, object]]] = {}

    cur_func = ""
    in_body = False
    cur_block = ""
    inst_seq = 0

    def resolve_scope_file(scope_id: Optional[int]) -> str:
        seen: Set[int] = set()
        cur = scope_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            file_id = scope_file_refs.get(cur)
            if file_id is not None:
                return files_by_id.get(file_id, default_file)
            cur = scope_parent_refs.get(cur)
        return default_file

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            meta = line.strip()
            m_file = _DIFILE_RE.match(meta)
            if m_file:
                filename = m_file.group(2)
                directory = m_file.group(3)
                file_path = (
                    filename
                    if os.path.isabs(filename)
                    else os.path.join(directory, filename)
                )
                files_by_id[int(m_file.group(1))] = file_path
                default_file = file_path
                continue

            m_loc_scope = _DILOC_SCOPE_RE.match(meta)
            if m_loc_scope:
                line_n = int(m_loc_scope.group(2))
                col_n = (
                    int(m_loc_scope.group(3))
                    if m_loc_scope.group(3) is not None
                    else 0
                )
                dilocs[int(m_loc_scope.group(1))] = (
                    line_n,
                    col_n,
                    int(m_loc_scope.group(4)),
                )
                continue

            m_loc = _DILOC_RE.match(meta)
            if m_loc:
                line_n = int(m_loc.group(2))
                col_n = int(m_loc.group(3)) if m_loc.group(3) is not None else 0
                dilocs[int(m_loc.group(1))] = (line_n, col_n, None)
                continue

            m_file_ref = _DIMETA_FILE_REF_RE.match(meta)
            if m_file_ref:
                scope_file_refs[int(m_file_ref.group(1))] = int(m_file_ref.group(2))
            m_scope_ref = _DIMETA_SCOPE_REF_RE.match(meta)
            if m_scope_ref:
                scope_parent_refs[int(m_scope_ref.group(1))] = int(
                    m_scope_ref.group(2)
                )

            if line.startswith("---"):
                in_body = False
                cur_func = ""
                cur_block = ""
                inst_seq = 0
                continue

            if line.startswith("name:"):
                cur_func = line.split(":", 1)[1].strip().strip("'\"")
                rows_by_func.setdefault(cur_func, [])
                continue

            if line.startswith("body:") and cur_func:
                in_body = True
                cur_block = ""
                inst_seq = 0
                continue

            if not in_body or not cur_func:
                continue

            stripped = line.strip()
            if not stripped or stripped in {"|", "..."}:
                continue
            if stripped.endswith(":") and stripped.startswith("bb."):
                cur_block = stripped[:-1]
                inst_seq = 0
                continue
            if (
                not cur_block
                or stripped.startswith(("liveins:", "successors:", ";", "- "))
            ):
                continue

            loc_file = "<synthetic>"
            loc_line = 0
            loc_col = 0
            m_dbg = _MIR_DBG_REF_RE.search(stripped)
            if m_dbg:
                loc_line, loc_col, loc_scope = dilocs.get(
                    int(m_dbg.group(1)), (0, 0, None)
                )
                loc_file = resolve_scope_file(loc_scope)
                stripped = _MIR_DBG_REF_RE.sub("", stripped).rstrip(" ,")

            rows_by_func[cur_func].append(
                {
                    "basicblock": cur_block,
                    "inst_seq": inst_seq,
                    "opcode": _mir_opcode(stripped),
                    "inst_text": stripped,
                    "file_path": loc_file,
                    "line": loc_line,
                    "col": loc_col,
                }
            )
            inst_seq += 1

    return rows_by_func


def _render_function_page(
    function: str,
    file_path: str,
    page: str,
    funcs: Sequence[Tuple[str, str, int, int, str]],
    source_dirs: Sequence[str],
    line_pages: Dict[int, str],
    initial_ir_rows: Sequence[sqlite3.Row],
    initial_ir_seq: int,
    final_mir_rows: Sequence[sqlite3.Row],
    final_mir_seq: int,
    history: Dict[str, List[Dict[str, object]]],
    assembly_page: str = "",
    final_mir_label: str = "",
) -> str:
    # Locations shared by initial IR and final-stage MIR get a stable per-loc
    # background so the cross-stage mapping is visible at a glance.
    initial_locs = {l for l in (_row_loc(r) for r in initial_ir_rows) if l}
    final_locs = {l for l in (_row_loc(r) for r in final_mir_rows) if l}
    shared_locs: FrozenSet[str] = frozenset(initial_locs & final_locs)
    func_list = _render_func_list(funcs, function)
    source = _render_source_panel(file_path, source_dirs, line_pages)
    initial = (
        _render_inst_panel(
            initial_ir_rows, f"initial IR: seq={initial_ir_seq}", shared_locs
        )
        if initial_ir_rows and initial_ir_seq >= 0
        else '<div class="empty">No initial IR rows recorded for this function.</div>'
    )
    final = (
        _render_inst_panel(
            final_mir_rows,
            final_mir_label or f"final MIR: seq={final_mir_seq}",
            shared_locs,
        )
        if final_mir_rows and final_mir_seq >= 0
        else '<div class="empty">No final MIR rows recorded for this function.</div>'
    )
    hist_json = json.dumps(history, separators=(",", ":"))
    asm_link = (
        f" &nbsp; <a href='{_esc(assembly_page)}'>assembly</a>"
        if assembly_page
        else ""
    )
    history_empty = (
        '<div class="empty">Click an instruction in any snapshot panel to '
        "see its IR/MIR evolution history.</div>"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{_esc(function)}</title>"
        "<link rel='stylesheet' href='style.css'>"
        f"<script>{_SCRIPT_JS}</script>"
        "</head><body>"
        f"<header><a href='index.html'>&larr; index</a> &nbsp; "
        f"<b>{_esc(function)}</b> "
        f"<span class='loc'>{_esc(file_path)}</span>{asm_link}</header>"
        "<div class='layout'>"
        f"{_render_panel('funcs', 'Functions', func_list)}"
        f"{_render_splitter('funcs', 'source')}"
        f"{_render_panel('source', 'Source', source)}"
        f"{_render_splitter('source', 'initial')}"
        f"{_render_panel('initial', 'Initial IR', initial)}"
        f"{_render_splitter('initial', 'final')}"
        f"{_render_panel('final', 'Final MIR', final)}"
        f"{_render_splitter('final', 'history')}"
        f"{_render_panel('history', 'IR/MIR Evolution History', history_empty)}"
        "</div>"
        f"<script>window.CURRENT_PAGE={json.dumps(page)};window.HIST={hist_json};</script>"
        "</body></html>"
    )


def _render_index(
    funcs: Sequence[Tuple[str, str, int, int, str]],
    pass_count: int,
    inst_count: int,
    assembly_page: str = "",
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
    asm_link = (
        f" &nbsp; <a href='{_esc(assembly_page)}'>assembly</a>"
        if assembly_page
        else ""
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ir-tracker report</title>"
        "<link rel='stylesheet' href='style.css'></head><body>"
        "<header>ir-tracker report &mdash; "
        f"{pass_count} pass snapshots, {inst_count} instruction rows, "
        f"{len(funcs)} functions{asm_link}</header>"
        "<div style='padding:10px'>"
        f"<table class='idx'>{''.join(rows)}</table>"
        "</div></body></html>"
    )


def _render_assembly_page(assembly_path: str) -> str:
    with open(assembly_path, "r", encoding="utf-8", errors="replace") as f:
        assembly = f.read()
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>assembly</title>"
        "<link rel='stylesheet' href='style.css'></head><body>"
        "<header><a href='index.html'>&larr; index</a> &nbsp; "
        f"<b>assembly</b> <span class='loc'>{_esc(assembly_path)}</span></header>"
        f"<pre class='asm'>{_esc(assembly)}</pre>"
        "</body></html>"
    )


def generate_html(
    con: sqlite3.Connection,
    output_dir: str,
    source_dirs: Sequence[str],
    all_passes: bool,
    no_highlight: bool,
    file_filter: str = "",
    assembly_path: str = "",
    mir_path: str = "",
) -> int:
    # ``all_passes`` and ``no_highlight`` are accepted for CLI compatibility
    # but unused in the function-centric layout (always emits per-location
    # dedup, no syntax highlighting).
    del all_passes, no_highlight

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
    if assembly_path and not os.path.isfile(assembly_path):
        print(f"ir-tracker: assembly file not found: {assembly_path}", file=sys.stderr)
        return 1
    if mir_path and not os.path.isfile(mir_path):
        print(f"ir-tracker: MIR file not found: {mir_path}", file=sys.stderr)
        return 1

    pass_count = int(
        con.execute(
            f"SELECT COUNT(*) AS c FROM {irtrackdb.T_PASSES} "
            f"WHERE kind IN ('ir', 'mir')"
        ).fetchone()["c"]
    )
    inst_count = int(
        con.execute(
            f"SELECT COUNT(*) AS c FROM {irtrackdb.T_INSTR} i "
            f"JOIN {irtrackdb.T_PASSES} p ON i.pass_id = p.id "
            f"WHERE p.kind IN ('ir', 'mir')"
        ).fetchone()["c"]
    )
    assembly_page = "assembly.html" if assembly_path else ""
    final_mir_by_func = parse_mir_snapshots(mir_path) if mir_path else {}
    page_by_function = {fn: page for fn, _fpath, _ni, _np, page in funcs}
    line_pages_by_file = _source_line_pages(con, page_by_function)

    with open(os.path.join(output_dir, "style.css"), "w", encoding="utf-8") as f:
        f.write(_STYLE_CSS)

    for function, file_path, _ni, _np, page in funcs:
        iseq = _initial_seq(con, function, "ir")
        initial = _snapshot_rows(con, function, "ir", iseq) if iseq >= 0 else []
        mfseq = _last_seq(con, function, "mir")
        if function in final_mir_by_func:
            mfinal = final_mir_by_func[function]
            final_mir_label = f"final MIR: {os.path.basename(mir_path)}"
            if mfseq < 0:
                mfseq = 0
        else:
            mfinal = _snapshot_rows(con, function, "mir", mfseq) if mfseq >= 0 else []
            final_mir_label = ""
        hist = _history(con, function)
        html_text = _render_function_page(
            function,
            file_path,
            page,
            funcs,
            source_dirs,
            {
                line: target_page
                for (path, line), target_page in line_pages_by_file.items()
                if path == file_path
            },
            initial,
            iseq,
            mfinal,
            mfseq,
            hist,
            assembly_page,
            final_mir_label,
        )
        with open(os.path.join(output_dir, page), "w", encoding="utf-8") as f:
            f.write(html_text)

    if assembly_path:
        with open(os.path.join(output_dir, assembly_page), "w", encoding="utf-8") as f:
            f.write(_render_assembly_page(assembly_path))

    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(_render_index(funcs, pass_count, inst_count, assembly_page))

    print(f"ir-tracker: wrote {len(funcs)} function page(s) + index to {output_dir}")
    return 0
