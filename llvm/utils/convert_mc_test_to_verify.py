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

import argparse
import os
import pathlib
import re
import shlex
import sys

from UpdateTestChecks import common

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lit"))
from lit.ShCommands import Pipeline
from lit.ShUtil import ShParser
from lit.TestRunner import (
    CommandDirective,
    IntegratedTestKeywordParser,
    ParserKind,
    kPdbgRegex,
    parseIntegratedTestScriptCommands,
)

TEST_SUFFIXES = {".s", ".S", ".asm", ".txt", ".test"}
# The body of a diagnostic check, e.g. ':[[@LINE-1]]:11: error: message'.
DIAG_RE = re.compile(
    r"""(?:(?:<stdin>|[^\s:\[]*):?(?:\[\[\#?\s*@LINE(?P<offset>[+-]\d+)?\]\]|\b\d+):
           (?:(?P<col>\d+):)?\s*)?
        (?:(?P<kind>error|warning|note|remark):\s*)?
        (?P<msg>.+?)(?:\{\{\$\}\})?""",
    re.VERBOSE,
)


def parse_run_lines(path):
    """Return (first line, last line, shell AST) for each RUN line in path."""
    parser = IntegratedTestKeywordParser("RUN:", ParserKind.COMMAND, initial_value=[])
    for line_number, _, line in parseIntegratedTestScriptCommands(str(path), ["RUN:"]):
        parser.parseLine(line_number, line)
    # Drop the '%dbg(RUN: at line N)' that lit prepends to each command.
    return [
        (
            d.start_line_number,
            d.end_line_number,
            ShParser(re.fullmatch(kPdbgRegex, d.command)[2]).parse(),
        )
        for d in parser.getValue()
        if isinstance(d, CommandDirective)
    ]


def is_negative_mc_run(ast):
    """Match 'not llvm-mc ... 2>&1 | FileCheck ...'."""
    if not isinstance(ast, Pipeline) or len(ast.commands) != 2:
        return False
    mc, filecheck = ast.commands
    return (
        mc.args[:2] == ["not", "llvm-mc"]
        and ((">&", 2), "1") in mc.redirects
        and filecheck.args[0] == "FileCheck"
    )


def prefixes(command):
    return common.get_check_prefixes(shlex.join(command.args))


def commands(ast):
    if isinstance(ast, Pipeline):
        yield from ast.commands
    else:
        yield from commands(ast.lhs)
        yield from commands(ast.rhs)


def to_shell(command):
    """Print a parsed command without its 2>&1 redirect."""
    words = [
        arg if re.fullmatch(r"[\w@%+=:,./-]+", arg) else shlex.quote(arg)
        for arg in command.args
    ]
    for (op, *fd), arg in command.redirects:
        if (op, *fd) != (">&", 2):
            words.append(f"{fd[0] if fd else ''}{op} {arg}")
    return " ".join(words)


def find_code_line(lines, i):
    """Return the offset from line i to the closest code line, skipping checks."""
    for step in (-1, 1):
        j = i + step
        while 0 <= j < len(lines) and common.CHECK_RE.match(lines[j]):
            j += step
        line = lines[j].strip() if 0 <= j < len(lines) else ""
        if line and not line.startswith(("#", "//")):
            return f"@{j - i:+d}"
    return "@above"


def convert(path, text):
    try:
        runs = parse_run_lines(path)
    except ValueError:  # E.g. lit's %if substitutions.
        return None
    if any(
        cmd.args[0] == "split-file" or any("%t" in arg for arg in cmd.args)
        for _, _, ast in runs
        for cmd in commands(ast)
    ):
        return None
    neg = [prefixes(ast.commands[1]) for _, _, ast in runs if is_negative_mc_run(ast)]
    pos = {
        p
        for _, _, ast in runs
        if not is_negative_mc_run(ast)
        for cmd in commands(ast)
        if cmd.args[0] == "FileCheck"
        for p in prefixes(cmd)
    }
    neg_prefixes = {p for ps in neg for p in ps}
    if not neg or neg_prefixes & pos:
        return None

    # Prefixes used by every RUN line become 'expected', others e.g. 'rv32'.
    shared = set.intersection(*map(set, neg))
    verify_prefix = {
        p: "expected" if p in shared else re.sub(r"^check[-_]?", "", p.lower())
        for p in neg_prefixes
    }

    # Split on '\n' only, so that the line numbers match lit's.
    lines = text.split("\n")
    # Replace each converted (possibly multi-line) RUN line with a single line,
    # bottom-up so that the earlier line numbers stay valid.
    for first, last, ast in reversed(runs):
        if not is_negative_mc_run(ast):
            continue
        mc, filecheck = ast.commands
        names = sorted({verify_prefix[p] for p in prefixes(filecheck)})
        flag = "--verify" if names == ["expected"] else "--verify=" + ",".join(names)
        mc.args = ["llvm-mc", flag] + mc.args[2:]
        leader = lines[first - 1][: lines[first - 1].index("RUN:") + len("RUN:")]
        lines[first - 1 : last] = [f"{leader} {to_shell(mc)}"]

    prefix_alt = "|".join(map(re.escape, neg_prefixes))
    check_re = re.compile(
        rf"^(?P<code>.*?)(?P<comment>(?:#|//)\s*)(?P<prefix>{prefix_alt})"
        rf"(?P<suffix>-NEXT|-LABEL)?:\s*(?P<body>.*?)\s*$"
    )
    for i, line in enumerate(lines):
        m = check_re.match(line)
        if not m or common.RUN_LINE_RE.match(line):
            continue
        diag = DIAG_RE.fullmatch(m["body"])
        if not diag or m["suffix"] == "-LABEL" or (m["suffix"] and not diag["kind"]):
            return None
        if diag["offset"]:
            loc = "@" + diag["offset"]
        elif m["code"].strip():
            loc = ""  # Trailing comment on the same line as the code.
        else:
            loc = find_code_line(lines, i)
        if diag["col"]:
            loc += ":" + diag["col"]
        kind = (diag["kind"] or "error") + ("-re" if "{{" in diag["msg"] else "")
        directive = f"{verify_prefix[m['prefix']]}-{kind}{loc} {{{{{diag['msg']}}}}}"
        lines[i] = m["code"] + m["comment"] + directive
    return "\n".join(lines)


def find_tests(paths):
    for path in map(pathlib.Path, paths):
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.suffix in TEST_SUFFIXES)
        else:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "tests", nargs="+", help="test files or directories to convert in place"
    )
    for path in find_tests(parser.parse_args().tests):
        # Some test inputs are not valid UTF-8, so preserve the raw bytes.
        text = path.read_text(errors="surrogateescape")
        new_text = convert(path, text)
        if new_text is not None and new_text != text:
            path.write_text(new_text, errors="surrogateescape")
            print(f"Converted {path}")


if __name__ == "__main__":
    main()
