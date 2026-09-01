#!/usr/bin/env python3

# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks that files in selected source subtrees are Bazel target inputs."""

import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Optional, Set

BAZEL_REPOSITORY = "@llvm-project"
SUBTREES = [
    # TODO: Add more directories as we support most of their files in bazel.
    # "libc",
    # "lldb",
    "lld",
]
IGNORED_GLOBS = [
    "**/*.md",
    "**/*.txt",
    "**/*.TXT",  # Both cases are used
    "**/.clang-format",
    "**/.clang-tidy",
    "**/.gitignore",
    "**/cmake/**",
    "**/CMakeLists.txt",
    "**/docs/**",
    "**/utils/**",
    "clang/www/**",
    "lld/test/Unit/lit.cfg.py",  # cmake shims
    "lld/test/Unit/lit.site.cfg.py.in",  # cmake shims
]


def source_files(project_root: Path) -> Set[str]:
    files = set()
    for subtree in SUBTREES:
        subtree_root = project_root / subtree
        if not subtree_root.is_dir():
            raise RuntimeError(f"source subtree does not exist: {subtree_root}")
        for path in subtree_root.rglob("*"):
            relative_path = path.relative_to(project_root)
            source_path = relative_path.as_posix()
            if path.is_file() and not any(
                fnmatch(source_path, glob) for glob in IGNORED_GLOBS
            ):
                files.add(source_path)
    return files


def label_to_source_path(label: str) -> Optional[str]:
    repository, separator, remainder = label.partition("//")
    if not separator or repository != BAZEL_REPOSITORY:
        return None

    package, separator, target = remainder.partition(":")
    if not separator:
        return None
    return str(PurePosixPath(package, target))


def bazel_input_files(bazel_workspace: Path) -> Set[str]:
    target_patterns = " union ".join(
        f"{BAZEL_REPOSITORY}//{subtree}/..." for subtree in SUBTREES
    )
    query = f'kind("source file", deps({target_patterns}))'
    result = subprocess.run(
        ["bazel", "query", "--output=label", query],
        cwd=bazel_workspace,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode:
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"bazel query failed with exit code {result.returncode}")

    return {
        path
        for label in result.stdout.splitlines()
        if (path := label_to_source_path(label)) is not None
    }


def main() -> int:
    script = Path(__file__).resolve()
    bazel_workspace = script.parents[1]
    project_root = script.parents[3]

    try:
        files = source_files(project_root)
        orphaned = files - bazel_input_files(bazel_workspace)
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if orphaned:
        print("Files not used as inputs by any Bazel target:", file=sys.stderr)
        for path in sorted(orphaned):
            print(f"  {path}", file=sys.stderr)
        return 1

    subtree_word = "subtree" if len(SUBTREES) == 1 else "subtrees"
    print(f"All {len(files)} files in {len(SUBTREES)} {subtree_word} are Bazel inputs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
