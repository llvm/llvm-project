# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wrapper for generating compile_commands.json for LLVM Bazel build."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from python.runfiles import Runfiles

REQUIRED_EXTRA_AQUERY_ARGS = [
    "--noprocess_headers_in_dependencies",
    "--features=-parse_headers",
    "--host_features=-parse_headers",
    "--features=-layering_check",
    "--host_features=-layering_check",
]

OVERLAY_PACKAGES = {"", "tools", "llvm_configs", "examples"}


def _normalize_target(target: str) -> str:
    """Normalize user-friendly target names to @llvm-project form."""
    if target.startswith("@"):
        return target
    if target == "//...":
        return "@llvm-project//..."
    if "(" in target:
        return target
    if target.startswith("//"):
        pkg = target[2:].split("/")[0].split(":")[0]
        if pkg not in OVERLAY_PACKAGES:
            return f"@llvm-project{target}"
        return target
    if target.startswith(":"):
        return target
    pkg = target.split("/")[0].split(":")[0]
    if pkg not in OVERLAY_PACKAGES:
        return f"@llvm-project//{target}"
    return f"//{target}"


def _create_root_symlink(workspace_dir: Path) -> None:
    """Create a symlink to compile_commands.json at the git repo root if applicable."""
    src = workspace_dir / "compile_commands.json"
    if not src.is_file():
        return

    repo_root = workspace_dir.parent.parent
    if not ((repo_root / ".git").exists() or (repo_root / "llvm").is_dir()):
        return

    dest = repo_root / "compile_commands.json"
    try:
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        rel_target = os.path.relpath(src, repo_root)
        dest.symlink_to(rel_target)
        print(f"Created symlink: {dest} -> {rel_target}", file=sys.stderr)
    except OSError as err:
        print(f"Warning: Could not create root symlink at {dest}: {err}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="bazel run //tools/compile_commands --",
        description="Generate compile_commands.json for LLVM Bazel targets.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help=(
            "Bazel target(s) or query to generate compile commands for (default: @llvm-project//...). "
            "Shorthand like '//llvm:Support' is automatically resolved to '@llvm-project//llvm:Support'."
        ),
    )
    parser.add_argument(
        "--targets",
        "--target",
        dest="flag_targets",
        action="append",
        default=[],
        help="Target(s) to generate compile commands for (comma-separated or repeated).",
    )
    parser.add_argument(
        "--extra_aquery_arg",
        action="append",
        default=[],
        help="Extra argument to pass to aquery (can be used multiple times).",
    )
    parser.add_argument(
        "--exclude_system_include",
        action="append",
        default=[],
        help="Filter out '-isystem' arguments with this value (can be used multiple times).",
    )
    parser.add_argument(
        "--exclude_compile_arg",
        action="append",
        default=[],
        help="Filter out this compile argument from output (can be used multiple times).",
    )
    parser.add_argument(
        "--dump_aquery_output",
        help="Dump aquery output to this file.",
    )
    parser.add_argument(
        "--load_aquery_output",
        help="Load aquery output from this file.",
    )
    parser.add_argument(
        "--proto_type",
        choices=("proto", "jsonproto"),
        help="aquery output format (proto or jsonproto).",
    )

    args, unknown = parser.parse_known_args()

    r = Runfiles.Create()
    generator = r.Rlocation("wolfd_bazel_compile_commands/generate_compile_commands")
    if not generator or not os.path.isfile(generator):
        print(
            f"Error: Unable to locate generate_compile_commands binary in runfiles (path: {generator})",
            file=sys.stderr,
        )
        return 1

    # Normalize positional and flag targets
    targets = []
    for t in args.targets:
        targets.append(_normalize_target(t))
    for t in args.flag_targets:
        targets.extend(_normalize_target(item.strip()) for item in t.split(",") if item.strip())

    # Determine query expression
    if not targets:
        query = "@llvm-project//..."
    elif len(targets) == 1:
        query = targets[0]
    else:
        query = f"set({' '.join(targets)})"

    # Assemble generator command line
    cmd = [generator]
    for extra_arg in REQUIRED_EXTRA_AQUERY_ARGS:
        cmd.append(f"--extra_aquery_arg={extra_arg}")

    for extra_arg in args.extra_aquery_arg:
        cmd.append(f"--extra_aquery_arg={extra_arg}")

    for excl in args.exclude_system_include:
        cmd.append(f"--exclude_system_include={excl}")

    for excl in args.exclude_compile_arg:
        cmd.append(f"--exclude_compile_arg={excl}")

    if args.dump_aquery_output:
        cmd.extend(["--dump_aquery_output", args.dump_aquery_output])

    if args.load_aquery_output:
        cmd.extend(["--load_aquery_output", args.load_aquery_output])

    if args.proto_type:
        cmd.extend(["--proto_type", args.proto_type])

    cmd.extend(unknown)
    cmd.append(query)

    workspace_dir = Path(os.environ.get("BUILD_WORKSPACE_DIRECTORY", os.getcwd()))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as err:
        return err.returncode

    _create_root_symlink(workspace_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
