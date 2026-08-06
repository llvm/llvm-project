#!/usr/bin/env python3
"""Remove page-local contents directives from Sphinx documentation.

Furo provides an in-page navigation sidebar, so page-local contents blocks are
usually redundant. This helper keeps that removal mechanical and rerunnable.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def is_rst_contents(line: str) -> bool:
    return line.startswith(".. contents::")


def is_md_contents(line: str) -> bool:
    return line.startswith("```{contents}")


def remove_rst_contents(lines: list[str]) -> tuple[list[str], int]:
    result: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        if not is_rst_contents(lines[i]):
            result.append(lines[i])
            i += 1
            continue

        removed += 1
        i += 1
        while i < len(lines):
            line = lines[i]
            if line.startswith((" ", "\t")) or line.strip() == "":
                i += 1
                continue
            break

        while result and result[-1].strip() == "":
            result.pop()
        if i < len(lines) and result:
            result.append("\n")

    return result, removed


def remove_md_contents(lines: list[str]) -> tuple[list[str], int]:
    result: list[str] = []
    removed = 0
    i = 0
    while i < len(lines):
        if not is_md_contents(lines[i]):
            result.append(lines[i])
            i += 1
            continue

        removed += 1
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            i += 1
        if i < len(lines):
            i += 1

        while result and result[-1].strip() == "":
            result.pop()
        if i < len(lines) and result:
            result.append("\n")

    return result, removed


def rewrite(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if path.suffix in (".rst", ".td"):
        new_lines, removed = remove_rst_contents(lines)
    elif path.suffix == ".md":
        new_lines, removed = remove_md_contents(lines)
    else:
        return 0

    if removed:
        path.write_text("".join(new_lines), encoding="utf-8")
    return removed


def iter_sources(root: Path):
    if root.is_file():
        if root.suffix in (".rst", ".md", ".td"):
            yield root
        return

    for suffix in ("*.rst", "*.md", "*.td"):
        yield from root.rglob(suffix)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[
            Path("clang/docs"),
            Path("clang/Maintainers.md"),
            Path("clang/include/clang/Basic"),
            Path("clang/include/clang/Options"),
        ],
        help="Documentation roots to scan, defaults to Clang docs and generated-doc inputs.",
    )
    args = parser.parse_args()

    total = 0
    changed = 0
    for root in args.roots:
        for path in sorted(iter_sources(root)):
            removed = rewrite(path)
            if removed:
                changed += 1
                total += removed
                print(f"{path}: removed {removed} contents directive(s)")

    print(f"removed {total} contents directive(s) from {changed} file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
