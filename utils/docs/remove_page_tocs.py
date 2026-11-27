#!/usr/bin/env python3
"""Remove page-local contents directives from Sphinx documentation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

CONTENTS = re.compile(
    r"(?m)^```\{contents\}[^\n]*\n(?:.*\n)*?^```\n?|"
    r"\n*^\.\. contents::[^\n]*(?:\n(?:[ \t].*|[ \t]*))*\n*"
)
SUFFIXES = (".rst", ".md", ".td")


def iter_sources(root: Path):
    if root.is_file():
        if root.suffix in SUFFIXES:
            yield root
        return

    for suffix in SUFFIXES:
        yield from root.rglob(f"*{suffix}")


def rewrite(path: Path) -> int:
    text = path.read_text(encoding="utf-8")

    def replacement(match: re.Match[str]) -> str:
        if match.group(0).lstrip("\n").startswith("```") or not match.start():
            return ""
        return "\n\n" if match.end() < len(text) else "\n"

    new_text, removed = CONTENTS.subn(replacement, text)
    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
    return removed


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

    removals = [
        (path, rewrite(path))
        for root in args.roots
        for path in sorted(iter_sources(root))
    ]
    changed = [(path, count) for path, count in removals if count]
    for path, count in changed:
        print(f"{path}: removed {count} contents directive(s)")
    print(
        f"removed {sum(count for _, count in changed)} contents directive(s) from {len(changed)} file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
