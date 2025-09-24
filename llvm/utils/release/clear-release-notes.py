#!/usr/bin/env python3
# ===-- clear-release-notes.py  ---------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
#
# Clear release notes, which is needed when bumping trunk version.
#
# ===------------------------------------------------------------------------===#

import argparse
from pathlib import Path


def process_file(fpath: Path) -> None:
    # ReleaseNotes.rst/.md -> ReleaseNotesTemplate.txt
    template_path = fpath.with_name(f"{fpath.stem}Template.txt")
    fpath.write_text(template_path.read_text(), newline="\n")
    print(f"{fpath} updated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-root",
        default=None,
        help="LLVM source root (/path/llvm-project). Defaults to the "
        "llvm-project the script is located in.",
    )

    args = parser.parse_args()

    # Find llvm-project root
    source_root = Path(__file__).resolve().parents[3]

    if args.source_root:
        source_root = Path(args.source_root).resolve()

    files_to_update = (
        "clang/docs/ReleaseNotes.rst",
        "clang-tools-extra/docs/ReleaseNotes.rst",
        "flang/docs/ReleaseNotes.md",
        "lld/docs/ReleaseNotes.rst",
        "llvm/docs/ReleaseNotes.md",
    )

    for f in files_to_update:
        process_file(source_root / f)
