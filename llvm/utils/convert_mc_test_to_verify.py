#!/usr/bin/env python3
"""Convert 'not llvm-mc ... 2>&1 | FileCheck' tests to 'llvm-mc --verify'.

This only handles simple tests whose FileCheck lines already check every
diagnostic llvm-mc emits. Since --verify (unlike FileCheck) requires every
diagnostic to be matched, other tests will fail after conversion and need
the missing expected-* directives added by hand.

Tests are rewritten in place. Directories are searched recursively for test
files. Run the converted tests with llvm-lit afterwards and fix up or revert
any failures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import pathlib
import re
import shlex
import sys

from UpdateTestChecks import common

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "lit"))
from lit.ShCommands import Command, Pipeline
from lit.ShUtil import ShParser
from lit.TestRunner import CommandDirective, parseIntegratedTestScriptCommands

TEST_SUFFIXES = {".s", ".S", ".asm", ".txt", ".test"}
# The body of a diagnostic check, e.g. ':[[@LINE-1]]:11: error: message'.
DIAG_RE = re.compile(
    r"""
    (?:
      (?P<unknown><unknown>:0:\s*)
    |
      (?:<stdin>|[^\s:\[]*):?                       # Optional filename
      (?:
        \[\[\#?\s*@LINE(?P<offset>[+-]\d+)?\]\]     # [[@LINE]] or [[#@LINE+N]]
      | \[\[\#\]\]                                  # [[#]]
      | \{\{\[0-9\]\+\}\}                           # {{[0-9]+}}
      | \b\d+                                       # Fixed line number
      ):
      (?:
        (?:
          (?P<col>\d+)                              # Fixed column number
        | \[\[\#\]\]                                # [[#]]
        | \{\{[^}:\s]+\}\}                          # Regex column, e.g. {{[0-9]+}}
        ):
      )?\s*
    )?
    (?:(?P<kind>error|warning|note|remark):\s*)?
    (?P<msg>.+?)
    (?:\{\{\$\}\})?                                 # Optional end-of-line {{$}}
    """,
    re.VERBOSE,
)


@dataclass
class RunLine:
    first_line: int
    last_line: int
    mc: Command
    prefixes: list[str]


@dataclass
class Directive:
    indent: str
    comment: str
    prefix_kind: str
    col: str
    body: str

    def format(self, loc: str = "") -> str:
        return f"{self.comment}{self.prefix_kind}{loc}{self.col} {self.body}"


def parse_run_lines(path: pathlib.Path) -> list[RunLine]:
    """Return the 'not llvm-mc ... | FileCheck' RUN lines in path, or [] if the
    test has none, uses compound RUN lines (e.g. 'rm -rf %t && split-file %s %t'),
    or shares a FileCheck prefix with another RUN line."""
    directives: list[CommandDirective] = []
    for line_number, keyword, line in parseIntegratedTestScriptCommands(
        str(path), ["RUN:"]
    ):
        if directives and directives[-1].add_continuation(line_number, keyword, line):
            continue
        directives.append(CommandDirective(line_number, line_number, keyword, line))

    runs: list[RunLine] = []
    other_prefixes: set[str] = set()
    for d in directives:
        try:
            ast = ShParser(d.command).parse()
        except ValueError:  # E.g. lit's %if substitutions.
            return []
        if not isinstance(ast, Pipeline):
            return []

        prefixes: list[str] = []
        for cmd in ast.commands:
            if cmd.args[0] == "FileCheck":
                prefixes.extend(common.get_check_prefixes(shlex.join(cmd.args)))

        if is_negative_mc_run(ast):
            mc = ast.commands[0]
            runs.append(RunLine(d.start_line_number, d.end_line_number, mc, prefixes))
        else:
            other_prefixes.update(prefixes)

    for r in runs:
        if other_prefixes.intersection(r.prefixes):
            return []
    return runs


def is_negative_mc_run(ast: Pipeline) -> bool:
    """Match 'not llvm-mc ... 2>&1 | FileCheck ...'."""
    if len(ast.commands) != 2:
        return False
    mc, filecheck = ast.commands
    return (
        mc.args[:2] == ["not", "llvm-mc"]
        and ((">&", 2), "1") in mc.redirects
        # The directives must be in the input, so it has to be the test file.
        and ("%s" in mc.args or (("<",), "%s") in mc.redirects)
        and filecheck.args[0] == "FileCheck"
    )


def to_shell(command: Command) -> str:
    words = [shlex.quote(arg) for arg in command.args]
    # lit stores redirects as ((op,), target) or ((op, fd), target), e.g.
    # (("<",), "%s") for '< %s' and ((">", 2), "%t") for '2> %t'.
    for op, target in command.redirects:
        fd = str(op[1]) if len(op) > 1 else ""
        words.append(f"{fd}{op[0]} {target}")
    return " ".join(words)


def find_code_line(lines: list[str], i: int) -> int | None:
    """Return the offset from line i to the closest code line, skipping checks."""
    for step in (-1, 1):
        j = i + step
        while 0 <= j < len(lines) and common.CHECK_RE.match(lines[j]):
            j += step
        if not (0 <= j < len(lines)):
            continue
        line = lines[j].strip()
        if line and not line.startswith(("#", "//")):
            return j - i
    return None


def inline_directives(
    lines: list[str], standalone: dict[int, tuple[int, Directive]]
) -> list[str]:
    """Move standalone directives to their target code line (on the same line
    when there is only one, or immediately below as @-1, @-2, ...)."""
    by_target: dict[int, list[Directive]] = {}
    moved: set[int] = set()
    for i, (target, d) in standalone.items():
        if 0 <= target < len(lines) and target not in standalone:
            by_target.setdefault(target, []).append(d)
            moved.add(i)

    result: list[str] = []
    for i, line in enumerate(lines):
        if i in moved:
            continue
        if i not in by_target:
            result.append(line)
            continue

        directives = by_target[i]
        code = line.rstrip()
        d = directives[0]
        # Moving a standalone check onto a line shifts its EndOfStatement token
        # from '\n' (at len(code) + 1) to '#' (at len(code) + 2), so place
        # end-of-line column checks on the line below instead.
        at_eol = bool(d.col and int(d.col[1:]) > len(code))
        has_comment = "#" in code or "//" in code
        if (
            len(directives) == 1
            and not has_comment
            and not code.endswith("\\")
            and not at_eol
        ):
            result.append(f"{code} {d.format()}")
        else:
            result.append(line)
            for idx, d in enumerate(directives, start=1):
                result.append(f"{d.indent}{d.format(f'@-{idx}')}")
    return result


def convert(path: pathlib.Path, inline: bool = False) -> str | None:
    try:
        # Split on '\n' only, so that the line numbers match lit's.
        lines = path.read_text(encoding="utf-8").split("\n")
    except UnicodeDecodeError:
        return None

    runs = parse_run_lines(path)
    if not runs:
        return None

    neg_prefixes: set[str] = set()
    for r in runs:
        neg_prefixes.update(r.prefixes)

    verify_prefix: dict[str, str] = {}
    for p in neg_prefixes:
        used_by_all_runs = all(p in r.prefixes for r in runs)
        if used_by_all_runs:
            verify_prefix[p] = "expected"
        else:
            verify_prefix[p] = re.sub(r"^check[-_]+", "", p.lower())

    # Replace each converted (possibly multi-line) RUN line with a single line,
    # bottom-up so that the earlier line numbers stay valid.
    for r in reversed(runs):
        names = sorted({verify_prefix[p] for p in r.prefixes})
        flag = "--verify" if names == ["expected"] else "--verify=" + ",".join(names)
        r.mc.args = ["llvm-mc", flag] + r.mc.args[2:]
        r.mc.redirects.remove(((">&", 2), "1"))
        comment, _, _ = lines[r.first_line - 1].partition("RUN:")
        lines[r.first_line - 1 : r.last_line] = [f"{comment}RUN: {to_shell(r.mc)}"]

    prefix_alt = "|".join(map(re.escape, neg_prefixes))
    check_re = re.compile(
        rf"^(?P<code>.*?)(?P<comment>(?:#|//)\s*)(?P<prefix>{prefix_alt})"
        rf"(?P<suffix>-NEXT|-LABEL)?:\s*(?P<body>.*?)\s*$"
    )
    standalone: dict[int, tuple[int, Directive]] = {}
    for i, line in enumerate(lines):
        m = check_re.match(line)
        if not m or common.RUN_LINE_RE.match(line):
            continue
        if m["suffix"] == "-LABEL":
            return None
        diag = DIAG_RE.fullmatch(m["body"])
        if not diag:
            return None
        # A CHECK-NEXT line without a diagnostic kind is usually checking the
        # caret or source line printed after a diagnostic.
        if m["suffix"] == "-NEXT" and not diag["kind"]:
            return None

        offset = None
        if diag["unknown"]:
            loc = "@unknown"
        elif diag["offset"]:
            offset = int(diag["offset"])
            loc = "@" + diag["offset"]
        elif m["code"].strip():
            loc = ""  # Trailing comment on the same line as the code.
        else:
            offset = find_code_line(lines, i)
            loc = f"@{offset:+d}" if offset is not None else "@above"

        kind = diag["kind"] or "error"
        if "{{" in diag["msg"]:
            kind += "-re"
        d = Directive(
            indent=m["code"],
            comment=m["comment"],
            prefix_kind=f"{verify_prefix[m['prefix']]}-{kind}",
            col=":" + diag["col"] if diag["col"] else "",
            body="{{" + diag["msg"] + "}}",
        )
        lines[i] = m["code"] + d.format(loc)
        if not m["code"].strip() and offset is not None:
            standalone[i] = (i + offset, d)

    if inline:
        lines = inline_directives(lines, standalone)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="place single directives on the same line as the instruction and "
        "multiple directives immediately below (@-1, @-2, ...)",
    )
    parser.add_argument(
        "tests", nargs="+", help="test files or directories to convert in place"
    )
    args = parser.parse_args()
    for arg in args.tests:
        root = pathlib.Path(arg)
        paths = sorted(root.rglob("*")) if root.is_dir() else [root]
        for path in paths:
            if path.suffix not in TEST_SUFFIXES:
                continue
            new_text = convert(path, inline=args.inline)
            if new_text is not None:
                path.write_text(new_text, encoding="utf-8")
                print(f"Converted {path}")


if __name__ == "__main__":
    main()
