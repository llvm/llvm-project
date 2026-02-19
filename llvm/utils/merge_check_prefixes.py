#!/usr/bin/env python3
"""merge_check_prefixes.py – Merge redundant FileCheck prefixes.

Given a test file with N RUN lines, this script finds common subsequences between
those check blocks and merges them into a single block with fewer, semantically-
shared prefixes, often dramatically reducing file size. This is intended to
enable auto-fixing code where update_any_test_checks.py complains of conflicting
RUN lines, by recomputing a preferred new set.

With --split, this undoes the transform, expanding the file into a form that
update_any_test_checks.py can handle, while leaving behind some metadata so that
re-doing the transform generally assigns stable names.

Example intended workflow:

    # list of files for which there are conflicting prefixes to fix
    $ FILES=""

    # annotate each RUN line with a single separate check
    $ llvm/utils/merge_check_prefixes.py --split $FILES

    # run a tool to update each check
    $ build/bin/llvm-lit --update-tests -sv $FILES
    $ llvm/utils/update_any_test_checks.py $FILES

    # re-compute compact diff
    llvm/utils/merge_check_prefixes.py $FILES

Usage:
    merge_check_prefixes.py [--base-prefix BASE] [--dry-run] [--split] file [file ...]
"""

import argparse
import difflib
import os
import re
import sys

# ---------------------------------------------------------------------------
# Reuse helpers from UpdateTestChecks/common.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "UpdateTestChecks"))
import common  # noqa: E402  (after sys.path manipulation)

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Match check-prefix / check-prefixes flags in RUN lines.
_CHECK_PREFIX_FLAG_RE = re.compile(
    r"\s+(--?check-prefix(?:es)?[= ])(\S+)"
)

# Matches plain comment lines that act as separators (e.g. "//" or "//.").
# suffix=None is used as a sentinel for these in check-line tuples.
_PLAIN_COMMENT_RE = re.compile(r"^\s*(?://|;|#)\.?\s*$")

# Matches coverage comment lines written by --split mode, e.g.:
#   ; --check-prefix=CHECK-A=1, 2, 5
_COVERAGE_COMMENT_RE = re.compile(
    r"^\s*(?://|[;#])\s*--check-prefix=(\S+)=\s*([\d,\s]+)$"
)

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_coverage_comments(lines):
    """Re-compute prefix_name_map from coverage comment lines embedded in a file.

    Scans *lines* for lines matching the pattern written by --split mode:
        <comment> --check-prefix=NAME=idx, idx, ...
    Returns a dict mapping frozenset(rl_indices) -> prefix_name, or None if no
    such lines are found.
    """
    prefix_name_map = {}
    for line in lines:
        m = _COVERAGE_COMMENT_RE.match(line)
        if m:
            name = m.group(1)
            indices = frozenset(int(s) for s in m.group(2).split(',') if s.strip())
            prefix_name_map[indices] = name
    return prefix_name_map or None


def collect_check_lines_for_run(lines, prefixes):
    """Return (check_lines, line_numbers) for all *prefixes* of one RUN line.

    Iterates the file once in order. For each line, checks it against every
    prefix regex; if it matches, records it as a check line. Plain comment
    lines (e.g. "//" or "//.") that immediately follow a check line are also
    included.

    check_lines : list of (content, is_plain)
                  content is "suffix: body" for check lines, raw text for plain
                  is_plain is False for check lines, True for plain comments
    line_numbers: set of 0-based indices that belong to this run's check block
    """
    prefix_set = frozenset(prefixes)

    check_lines = []
    line_numbers = set()
    after_check = False  # True while we may still consume trailing plain comments

    for i, line in enumerate(lines):
        m = common.CHECK_RE.match(line)
        matched = m is not None and m.group(1) in prefix_set
        if matched:
            content = line[m.end(1):]
            check_lines.append((content, False, i))
            line_numbers.add(i)
            after_check = True
        if not matched:
            if after_check and _PLAIN_COMMENT_RE.match(line):
                check_lines.append((line, True, i))
                line_numbers.add(i)
            else:
                after_check = False

    return check_lines, line_numbers


def find_run_lines(test, lines):
    """Local copy of common.find_run_lines that also returns per-RUN line ranges.

    Returns (run_lines, run_line_ranges) where run_line_ranges is a list of
    range objects giving the 0-based line indices spanned by each RUN statement
    (accounting for backslash continuations).
    """
    common.debug("Scanning for RUN lines in test file:", test)
    raw = [(i, m.group(1))
           for i, l in enumerate(lines)
           for m in (common.RUN_LINE_RE.match(l),)
           if m]
    if not raw:
        return [], []

    run_lines = []
    run_line_ranges = []
    start_idx, content = raw[0]
    end_idx = start_idx
    for i, line_content in raw[1:]:
        if content.endswith("\\"):
            content = content.rstrip("\\") + " " + line_content
            end_idx = i
        else:
            run_lines.append(content)
            run_line_ranges.append(range(start_idx, end_idx + 1))
            start_idx = end_idx = i
            content = line_content
    run_lines.append(content)
    run_line_ranges.append(range(start_idx, end_idx + 1))

    common.debug("Found {} RUN lines in {}:".format(len(run_lines), test))
    for l in run_lines:
        common.debug("  RUN: {}".format(l))
    return run_lines, run_line_ranges


def longest_common_prefix(strings):
    if not strings:
        return ""
    s = strings[0]
    for t in strings[1:]:
        while not t.startswith(s):
            s = s[:-1]
            if not s:
                return ""
    return s


def auto_base_prefix(prefixes):
    """Detect a common leading string from all prefix names, e.g. 'CHECK'."""
    base = longest_common_prefix(list(prefixes))
    # Strip any trailing suffixes
    base = base.rstrip("0123456789-")
    return base or "CHECK"

# ---------------------------------------------------------------------------
# Score / distance
# ---------------------------------------------------------------------------

def _seq_to_strs(seq):
    return [(";" if c else "*") + content for content, c, _ in seq]


def score(seq_a, seq_b):
    """Number of diff lines (unified diff hunk lines) between two sequences."""
    a = _seq_to_strs(seq_a)
    b = _seq_to_strs(seq_b)
    return sum(1 for _ in difflib.unified_diff(a, b))

# ---------------------------------------------------------------------------
# Zipper merge
# ---------------------------------------------------------------------------

def zipper_merge(seq_a, seq_b):
    """Merge two check-line sequences using SequenceMatcher.

    Returns a new sequence of (suffix, content, present_in) triples where
    *present_in* is the union of the contributing sequences' present_in sets.
    Equal blocks are emitted once with a unioned present_in; A-only lines
    keep A's present_in; B-only lines keep B's present_in.  Within each
    "gap" between equal blocks, A-only lines come before B-only lines so
    that the logical order is: common, then A variant, then B variant.
    """
    a_strs = _seq_to_strs(seq_a)
    b_strs = _seq_to_strs(seq_b)
    sm = difflib.SequenceMatcher(None, a_strs, b_strs, autojunk=False)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ai, bi in zip(range(i1, i2), range(j1, j2)):
                ca, ia, pa = seq_a[ai]
                cb, ib, pb = seq_b[bi]
                result.append((ca, ia, pa | pb))
                assert(ca == cb)
                assert(ia == ib)
        elif tag == "replace":
            for k in range(i1, i2):
                result.append(seq_a[k])
            for k in range(j1, j2):
                result.append(seq_b[k])
        elif tag == "delete":
            for k in range(i1, i2):
                result.append(seq_a[k])
        elif tag == "insert":
            for k in range(j1, j2):
                result.append(seq_b[k])
    return result

# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------

def concat(items):
    """Concatenate all sequences in order without merging."""
    result = []
    for seq in items:
        result.extend(seq)
    return result

# ---------------------------------------------------------------------------
# Hierarchical (single-linkage) merge
# ---------------------------------------------------------------------------

def hierarchical_merge(items, label):
    """Single-linkage agglomerative merge of all sequences.

    *seqs* is a list of sequences; each line already carries its own pset.
    Returns a single merged sequence.
    """
    total = len(items) - 1
    step = 0
    prefix = f"{label} " if label else ""
    while len(items) > 1:
        n = len(items)
        best_score = None
        best_i = best_j = -1
        for i in range(n):
            for j in range(i + 1, n):
                s = score(items[i], items[j])
                if best_score is None or s < best_score:
                    best_score = s
                    best_i, best_j = i, j
        merged_seq = zipper_merge(items[best_i], items[best_j])
        # Remove j first (higher index) then i
        items.pop(best_j)
        items.pop(best_i)
        items.append(merged_seq)
        step += 1
        common.debug(f"  {prefix}merge {step}/{total} (score {best_score})", end="\r", flush=True)
    common.debug()

    return items[0]

# ---------------------------------------------------------------------------
# Prefix name assignment
# ---------------------------------------------------------------------------

def human_sort_key(s):
    """Sort key for alphanumeric strings, sorting digit runs by numeric value.

    E.g. "foo10" sorts after "foo9" rather than before it.
    """
    return [(0, int(p)) if p.isdigit() else (1, p) for p in re.split(r'(\d+)', s) if p]

def assign_prefix_names(block_merges, all_rls_frozenset, base, rl_coverage, run_prefix_map, all_prefixes, split):
    """Map each unique present_in set found in *block_merges* to a new prefix name.

    Each presence set is a frozenset of run-line strings.  *rl_coverage* maps
    frozenset(run_lines) → original_prefix_name for reuse when possible.
    *run_prefix_map* supplies the original prefix names used by each run line,
    for building generated names.
    """
    presence_sets = set()
    for seq in block_merges:
        for _content, _is_plain, pset in seq:
            presence_sets.add(pset)

    # Generate or reuse a new name from the distinguishing suffixes of the prefix
    # names used by the run lines in this set.
    def _pset_name(pset):
        if pset == all_rls_frozenset:
            return base
        existing = rl_coverage.get(pset)
        if existing is not None:
            return existing
        if split:
            # Create the name by combining all prefixes
            def _rl_suffixes_split(rl):
                """Extract ordered union of all parts from all prefixes for this run line."""
                result = {}
                for p in run_prefix_map[rl]:
                    result.update(dict.fromkeys(p.split("-")))
                result.pop("", None)
                if not result or (len(result) == 1 and base in result):
                    # if the only key is CHECK, convert to CHECK-1
                    result = dict.fromkeys([base, str(rl + 1)])
                return result
            suffix_sets = [_rl_suffixes_split(rl) for rl in sorted(pset)]
            # collect all unique suffixes in an ordered set
            all_unique = {}
            for s in suffix_sets:
                all_unique.update(s)
            keys = []
            # move base to the head, if present
            if base in all_unique:
                all_unique.pop(base)
                keys.append(base)
            keys.extend(all_unique.keys())
            candidate = "-".join(keys)
        else:
            # Create the name by finding any common part in each run line
            # and appending them together with other lines
            def _rl_suffixes_merge(rl):
                """Compute intersection of all parts across all prefixes for this run line."""
                suffix_sets = []
                for p in run_prefix_map[rl]:
                    suffix_sets.append(dict.fromkeys(p.split("-")))
                result = {}
                if suffix_sets:
                    result = suffix_sets[0]
                    for s in suffix_sets[1:]:
                        result = dict.fromkeys(k for k in result if k in s)
                    result.pop("", None)
                if not result or (len(result) == 1 and base in result):
                    # if the only key is CHECK, convert to CHECK-1
                    result = dict.fromkeys([base, str(rl + 1)])
                return result
            keys = {}
            for s in (_rl_suffixes_merge(rl) for rl in sorted(pset)):
                keys.update(s)
            candidate = []
            # move base to the head, if present
            if base in keys:
                keys.pop(base)
                candidate.append(base)
            candidate.extend(keys.keys())
            candidate = "-".join(candidate)
        name, n = candidate, 2
        while name in all_prefixes:
            name = candidate + "-_" + str(n)
            n += 1
        all_prefixes.add(name)
        return name

    return {pset: _pset_name(pset) for pset in presence_sets}

# ---------------------------------------------------------------------------
# RUN line rewriting
# ---------------------------------------------------------------------------

def new_prefixes_for_run(rl, prefix_name_map):
    """Return the sorted list of new prefix names that run line *rl* needs.

    A run line needs every new prefix whose presence set contains *rl*.
    """
    return sorted(name for pset, name in prefix_name_map.items() if rl in pset)


def rewrite_run_line(run_line, new_names, any_replaced=False):
    """Replace all --check-prefix(es) flags in *run_line* with the new list."""
    if len(new_names) == 1:
        replacement_flag = " --check-prefix=" + new_names[0]
    else:
        replacement_flag = " --check-prefixes=" + ",".join(new_names)

    # Replace the first occurrence; drop any subsequent check-prefix flags
    replaced = any_replaced

    is_default = replacement_flag == " --check-prefix=CHECK"

    def _replace(m):
        nonlocal replaced
        if not replaced:
            replaced = True
            return "" if is_default else replacement_flag
        return ""

    new_line = _CHECK_PREFIX_FLAG_RE.sub(_replace, run_line)
    return new_line, replaced

# ---------------------------------------------------------------------------
# File rewrite
# ---------------------------------------------------------------------------

def comment_char(lines):
    """Detect the comment character used in check lines."""
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("//"):
            return "//"
        if stripped.startswith(";"):
            return ";"
        if stripped.startswith("#"):
            return "#"
    return "//"


def emit_check_block(merged_seq, prefix_name_map, comment):
    """Render the merged check block as a list of text lines (with newlines)."""
    out = []
    for content, is_plain, pset in merged_seq:
        if is_plain:
            out.append(content)
        else:
            name = prefix_name_map[pset]
            out.append("{comment} {name}{content}".format(
                comment=comment, name=name, content=content))
    return out

# ---------------------------------------------------------------------------
# Main processing of a single file
# ---------------------------------------------------------------------------

def process_file(path, base_prefix_override=None, dry_run=False, split=False):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()

    # 1. Find RUN lines and extract FileCheck prefixes
    run_lines_raw, run_line_ranges = find_run_lines(path, lines)
    _filtered = []
    _filtered_ranges = []
    for rl, rl_range in zip(run_lines_raw, run_line_ranges):
        if "|" not in rl:
            common.warn("Skipping unparsable RUN line: " + rl)
        else:
            _filtered.append(rl)
            _filtered_ranges.append(rl_range)
    run_lines_raw = _filtered
    run_line_ranges = _filtered_ranges
    if not run_lines_raw:
        common.warn(f"{path}: no RUN lines found, skipping.")
        return

    # Map each run line to its FileCheck prefix(es)
    run_prefix_list = []   # ordered unique run-line strings
    _rl_to_idx = {}        # run_line_str -> int index into run_prefix_list
    run_prefix_map = []    # list of prefix lists, indexed by rl_idx
    all_prefixes = set()
    for rl in run_lines_raw:
        _rl_to_idx[rl] = len(run_prefix_list)
        run_prefix_list.append(rl)
        prefixes = common.get_check_prefixes(rl)
        run_prefix_map.append(prefixes)
        all_prefixes.update(prefixes)

    if not all_prefixes:
        common.warn(f"{path}: no FileCheck prefixes found, skipping.")
        return

    # 2. Collect check lines for each RUN line
    _seq_lnums = [collect_check_lines_for_run(lines, pfxs) for pfxs in run_prefix_map]
    run_seqs = [seq for seq, _ in _seq_lnums]
    check_line_indices = set().union(*(lnums for _, lnums in _seq_lnums))
    check_line_blocks = []
    for idx in range(len(lines)):
        if idx in check_line_indices:
            if check_line_blocks and check_line_blocks[-1].stop == idx:
                check_line_blocks[-1] = range(check_line_blocks[-1].start, idx + 1)
            else:
                check_line_blocks.append(range(idx, idx + 1))
    erase_line_indices = check_line_indices
    erase_line_indices.update(
        i for i, line in enumerate(lines) if _COVERAGE_COMMENT_RE.match(line)
    )

    # 3. Merge or split content
    common.debug(f"{path}: merging {len(run_seqs)} RUN-line sequences …")
    block_merges = []
    for blk_idx, blk in enumerate(check_line_blocks):
        seqs_to_merge = [
            [(content, is_plain, frozenset({rl_idx}))
             for content, is_plain, lnum in seq
             if lnum in blk]
            for rl_idx, seq in enumerate(run_seqs)
        ]
        seqs_to_merge = [s for s in seqs_to_merge if s]
        assert(seqs_to_merge)
        label = f"block {blk_idx + 1}/{len(check_line_blocks)}"
        block_merges.append(
            concat(seqs_to_merge) if split else hierarchical_merge(seqs_to_merge, label)
        )
    # 4. Assign new prefix names
    all_rls_fs = frozenset(range(len(run_seqs)))
    # Determine base prefix name
    base = base_prefix_override or auto_base_prefix(all_prefixes)
    # Map: frozenset(run_line_indices_using_p) → original prefix name.
    rl_coverage = {
        frozenset(rl_idx for rl_idx, pfxs in enumerate(run_prefix_map) if p in pfxs): p
        for p in all_prefixes
    }
    # If the file already contains coverage comments from a prior --split run,
    # fold their names into rl_coverage so assign_prefix_names will reuse them.
    coverage = parse_coverage_comments(lines)
    if coverage:
        rl_coverage.update(coverage)
        all_prefixes.update(coverage.values())
    prefix_name_map = assign_prefix_names(block_merges, all_rls_fs, base, rl_coverage, run_prefix_map, all_prefixes, split)

    # Print summary
    common.debug(f"  Base prefix: {base!r}")
    common.debug(f"  New prefixes ({len(prefix_name_map)}):")
    for pset, name in sorted(prefix_name_map.items(), key=lambda kv: kv[1]):
        members = ", ".join(str(rl_idx) for rl_idx in sorted(pset))
        common.debug(f"    {name!r}  ← {{{members}}}")
    merged_lines = sum(len(blk) for blk in block_merges)
    original_lines = sum(len(seq) for seq in run_seqs)
    common.debug(f"  Check lines: {original_lines} → {merged_lines} "
          f"({100*merged_lines//original_lines if original_lines else 0}%)")

    # 6. Rewrite RUN lines
    # Build a map: original_line_index -> new line text
    run_line_replacements = {}  # line_index -> new_text
    for rl_idx, rl_range in enumerate(run_line_ranges):
        if not run_seqs[rl_idx]:
            continue
        new_names = new_prefixes_for_run(rl_idx, prefix_name_map)
        any_replaced = False
        for i in rl_range:
            new_line, any_replaced = rewrite_run_line(lines[i], new_names, any_replaced)
            run_line_replacements[i] = new_line
        if not any_replaced:
            # No existing --check-prefix flag found; insert one after 'FileCheck'
            flag = ("--check-prefix=" + new_names[0] if len(new_names) == 1
                    else "--check-prefixes=" + ",".join(new_names))
            if flag != "--check-prefix=CHECK":
                for i in rl_range:
                    new_line = re.sub(r'\bFileCheck\b', 'FileCheck ' + flag, lines[i], count=1)
                    if new_line != lines[i]:
                        run_line_replacements[i] = new_line
                        break
                else:
                    common.warn(f"couldn't insert {flag!r} into RUN line", test_file=path)

    # 7. Build new file content
    comment = comment_char(lines)
    new_check_blocks = [emit_check_block(blk, prefix_name_map, comment) for blk in block_merges]
    blk_start_map = {blk.start: new_check_blocks[i] for i, blk in enumerate(check_line_blocks)}

    new_lines = []
    for i, line in enumerate(lines):
        if i in erase_line_indices:
            # Drop old check lines; insert merged block at the start of each original block
            if i in blk_start_map:
                new_lines.extend(blk_start_map[i])
            continue
        if i in run_line_replacements:
            new_lines.append(run_line_replacements[i])
        else:
            new_lines.append(line)

    # In split mode, insert a comment block before the first RUN line recording
    # the mapping of each new prefix name to the run-line indices that use it.
    if split:
        first_run = next(
            (j for j, ln in enumerate(new_lines) if common.RUN_LINE_RE.match(ln)),
            0
        )
        coverage_comments = [
            f"{comment} --check-prefix={name}={', '.join(str(i) for i in sorted(pset))}\n"
            for pset, name in sorted(rl_coverage.items(), key=lambda kv: human_sort_key(kv[1]))
        ]
        new_lines[first_run:first_run] = coverage_comments

    if dry_run:
        common.debug("  (dry-run: not modifying file)")
        return

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(new_lines)

    common.debug(f"  Written: {path}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge redundant FileCheck prefix blocks in LLVM test files."
    )
    parser.add_argument(
        "files", metavar="file", nargs="+", help="Test file(s) to process."
    )
    parser.add_argument(
        "--base-prefix",
        default=None,
        help="Override the auto-detected base prefix name (e.g. CHECK).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Concatenate check blocks in order instead of merging common lines.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show verbose output"
    )
    args = parser.parse_args()
    common._verbose = args.verbose

    for path in args.files:
        if not os.path.isfile(path):
            common.warn(f"{path!r} is not a file.")
            continue
        try:
            process_file(path, base_prefix_override=args.base_prefix,
                         dry_run=args.dry_run, split=args.split)
        except Exception as exc:
            common.warn(f"Error processing {path!r}: {exc}")
            raise


if __name__ == "__main__":
    main()
