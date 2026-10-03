#!/usr/bin/env python3
#
# ===----------------------------------------------------------------------------===#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------------===#

import argparse
import logging
import os
import re
import subprocess
import sys

from build import build_llvm
from build import ensure_llvm_tools
from lit import find_lit_tests
from patch import extract_modified_source_lines_from_patch
from patch import extract_source_files_from_patch
from patch import create_patch_from_last_commits
from patch import write_source_file_allowlist
from print import report_covered_and_uncovered_lines
from process import process_coverage_data
from test import run_modified_lit_tests
from test import run_modified_unit_tests
from utils import configure_logging
from utils import classify_tests
from utils import delete_profraw
from utils import log
from utils import mark_build_success
from utils import resolve_projects
from utils import should_rebuild
from utils import target_name

sys.path.append(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        prog="git patch-coverage",
        description="Patch-based code coverage tool for LLVM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-b",
        "--build-dir",
        help="path to build directory",
        dest="build_dir",
        default="build",
    )

    parser.add_argument(
        "-i",
        "--instrumented-build-dir",
        help="path to directory in which the tool will build projects with intrumentation enabled",
        dest="inst_build_dir",
        default="build_inst",
    )

    parser.add_argument(
        "-n",
        "--num-commits",
        help="number of commits to include in patch",
        dest="num_commits",
        default=1,
    )

    parser.add_argument(
        "binary",
        help="target binary to generate coverage for (e.g. opt, clang)",
    )

    parser.add_argument(
        "test_path",
        nargs="*",
        help="path to test suite/s (default: <build-dir>/test)",
        default=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--projects",
        help="LLVM projects to enable (semicolon-separated, e.g. clang;mlir)",
    )

    args = parser.parse_args()

    if not hasattr(args, "test_path") or not args.test_path:
        args.test_path = [os.path.join(args.build_dir, "test")]

    return (
        args.build_dir,
        args.inst_build_dir,
        args.num_commits,
        args.binary,
        args.test_path,
        args.projects,
    )


def main():
    (
        build_dir,
        inst_build_dir,
        num_commits,
        binary,
        test_paths,
        projects,
    ) = parse_args()

    configure_logging(inst_build_dir)
    projects = resolve_projects(projects, build_dir)

    # Ensure we have required tools to parse test suite info.
    ensure_llvm_tools(build_dir, projects, binary)

    # Create a diff file from the commit/s.
    patch_path = os.path.join(build_dir, "patch.diff")
    create_patch_from_last_commits(patch_path, num_commits)
    source_files = extract_source_files_from_patch(patch_path)

    # Get all the modified lines of patch, from both source files and test files.
    llvm_lit_path = os.path.join(build_dir, "bin/llvm-lit")
    tests = frozenset(find_lit_tests(llvm_lit_path, test_paths))
    modified_lines = extract_modified_source_lines_from_patch(patch_path, tests)

    # Use allow list feature to generate ".profraw" data for only source files in the patch.
    os.makedirs(inst_build_dir, exist_ok=True)
    allowlist_path = os.path.join(inst_build_dir, "fun.list")
    write_source_file_allowlist(source_files, allowlist_path)

    # Print all the modified lines of the patch.
    for file, lines in modified_lines.items():
        log(f"File: {os.path.relpath(file)[2:]}")
        for line_number, line_content in lines:
            cleaned_line_content = line_content.rstrip("\n")
            log(f"Line {line_number}: {cleaned_line_content}")
        log("")

    # Contruct the absolute path of binary that we want to check coverage for.
    unit_tests, lit_tests = classify_tests(patch_path)
    unit_binary = None
    lit_binary = None
    if lit_tests:
        lit_binary = os.path.abspath(os.path.join(inst_build_dir, "bin", binary))
    if unit_tests:
        target = target_name(patch_path, inst_build_dir)
        unit_binary = os.path.abspath(os.path.join(inst_build_dir, target))

    # Build the LLVM in instrumented directory using LLVM_BUILD_INSTRUMENTED_COVERAGE.
    rebuild = should_rebuild(inst_build_dir, patch_path, lit_binary or unit_binary)
    if rebuild:
        delete_profraw(inst_build_dir)
        build_llvm(inst_build_dir, build_dir, binary, projects, allowlist_path)
        mark_build_success(inst_build_dir, patch_path)
    else:
        print("\n[patch-coverage] Skipping patch coverage (no changes)")
        sys.exit(0)

    # Run all the test cases of patch with instrumented binary.
    inst_lit_path = os.path.join(inst_build_dir, "bin/llvm-lit")
    patch_path = os.path.abspath(patch_path)
    run_modified_lit_tests(inst_lit_path, patch_path, tests, inst_build_dir)
    run_modified_unit_tests(build_dir, inst_build_dir, patch_path)

    # Report covered and uncovered lines of each source file.
    coverage_files = process_coverage_data(
        source_files,
        inst_build_dir,
        lit_binary,
        unit_binary,
        patch_path,
    )
    report_covered_and_uncovered_lines(coverage_files, modified_lines)

    # Remove redundant "default.profraw" generated in source root.
    curr_dir = os.path.dirname(os.getcwd())
    default_profraw_path = os.path.join(curr_dir, "default.profraw")
    try:
        os.remove(default_profraw_path)
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
