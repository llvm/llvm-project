#!/usr/bin/env python3
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

"""
Run idt on the headers changed in a PR and (optionally) post the resulting
LLVM_ABI annotation diff as a PR comment.

The heavy lifting (per-category clang flags, system-header-prefix isolation,
output filtering, the `hasBody` / out-of-line distinction) lives in
`llvm/utils/idt/idt.cc`. This script is a thin orchestrator: it picks which
headers to check, finds a translation unit for each (since LLVM's compile
database doesn't list headers), and forwards the list to idt batched by
category.

Usage as a git pre-commit hook:

    ln -s $(pwd)/llvm/utils/git/ids-check-helper.py .git/hooks/pre-commit

Environment variables `IDT_PATH` and `COMPILE_COMMANDS_PATH` may be used in
place of the matching command-line arguments.
"""


# Headers we skip outright. Each entry is matched against the changed-file
# path with `startswith`, so it can be either a directory prefix (trailing /)
# or a specific file path.
SKIP_HEADERS = [
    # PDB DIA requires Windows ATL (atlbase.h), unbuildable on Linux runners.
    # The direct mapping picks lib/DebugInfo/PDB/DIA/*.cpp which can't parse.
    "llvm/include/llvm/DebugInfo/PDB/DIA/",
    # No in-tree non-tools/non-target source #includes these headers, so we
    # can't run idt on a TU that pulls them in.
    "llvm/include/llvm/ExecutionEngine/Interpreter.h",  # only lli.cpp
    "llvm/include/llvm/Support/DebugLog.h",             # only lib/Target/RISCV/
    "llvm/include/llvm/Support/TargetSelect.h",         # only target-registration sources
]


# Manual header -> source mappings for headers that don't fit the conventional
# `llvm/include/llvm/<Subsystem>/<Bar>.h` -> `llvm/lib/<Subsystem>/<Bar>.cpp`
# layout. Used when neither the direct mapping nor the same-subdirectory
# grep fallback finds a workable source.
HEADER_SOURCE_OVERRIDES = {
    # LLVM-C headers
    "llvm/include/llvm-c/Analysis.h":          "llvm/lib/Analysis/Analysis.cpp",
    "llvm/include/llvm-c/BitReader.h":         "llvm/lib/Bitcode/Reader/BitReader.cpp",
    "llvm/include/llvm-c/BitWriter.h":         "llvm/lib/Bitcode/Writer/BitWriter.cpp",
    "llvm/include/llvm-c/Comdat.h":            "llvm/lib/IR/Comdat.cpp",
    "llvm/include/llvm-c/Core.h":              "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm-c/DebugInfo.h":         "llvm/lib/IR/DebugInfo.cpp",
    "llvm/include/llvm-c/Disassembler.h":      "llvm/lib/MC/MCDisassembler/Disassembler.cpp",
    "llvm/include/llvm-c/ErrorHandling.h":     "llvm/lib/Support/ErrorHandling.cpp",
    "llvm/include/llvm-c/ExecutionEngine.h":   "llvm/lib/ExecutionEngine/ExecutionEngineBindings.cpp",
    "llvm/include/llvm-c/IRReader.h":          "llvm/lib/IRReader/IRReader.cpp",
    "llvm/include/llvm-c/LLJIT.h":             "llvm/lib/ExecutionEngine/Orc/LLJITUtilsCBindings.cpp",
    "llvm/include/llvm-c/LLJITUtils.h":        "llvm/lib/ExecutionEngine/Orc/Debugging/LLJITUtilsCBindings.cpp",
    "llvm/include/llvm-c/Linker.h":            "llvm/lib/Linker/LinkModules.cpp",
    "llvm/include/llvm-c/Object.h":            "llvm/lib/Object/Object.cpp",
    "llvm/include/llvm-c/Orc.h":               "llvm/lib/ExecutionEngine/Orc/OrcV2CBindings.cpp",
    "llvm/include/llvm-c/OrcEE.h":             "llvm/lib/ExecutionEngine/Orc/OrcV2CBindings.cpp",
    "llvm/include/llvm-c/Remarks.h":           "llvm/lib/Remarks/RemarkParser.cpp",
    "llvm/include/llvm-c/Support.h":           "llvm/lib/Support/CommandLine.cpp",
    "llvm/include/llvm-c/Target.h":            "llvm/lib/Target/Target.cpp",
    "llvm/include/llvm-c/TargetMachine.h":     "llvm/lib/Target/TargetMachineC.cpp",
    "llvm/include/llvm-c/Transforms/PassBuilder.h": "llvm/lib/Passes/PassBuilderBindings.cpp",
    "llvm/include/llvm-c/Types.h":             "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm-c/lto.h":               "llvm/tools/lto/lto.cpp",

    # Top-level llvm/ headers
    "llvm/include/llvm/Pass.h":                "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/PassAnalysisSupport.h": "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/PassRegistry.h":        "llvm/lib/IR/PassRegistry.cpp",
    "llvm/include/llvm/PassSupport.h":         "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/InitializePasses.h":    "llvm/lib/Analysis/Analysis.cpp",
    "llvm/include/llvm/LinkAllIR.h":           "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm/LinkAllPasses.h":       "llvm/lib/IR/Pass.cpp",

    # Headers under llvm/include/llvm/Target/ whose same-subdir grep fallback
    # picks per-target sources (e.g. X86, AArch64). Redirect to lib/CodeGen/.
    "llvm/include/llvm/Target/CGPassBuilderOption.h": "llvm/lib/CodeGen/TargetPassConfig.cpp",
    "llvm/include/llvm/Target/TargetOptions.h":       "llvm/lib/CodeGen/AsmPrinter/AsmPrinter.cpp",

    # Headers pulled in transitively by a small set of "umbrella" sources.
    "llvm/include/llvm/ADT/ilist_node_base.h":                    "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/Analysis/SimplifyQuery.h":                 "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/CodeGenTypes/MachineValueType.h":          "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/Analysis.h":                            "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/ConstantFolder.h":                      "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/FMF.h":                                 "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/GenericFloatingPointPredicateUtils.h":  "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/IRBuilderFolder.h":                     "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/Support/Recycler.h":                       "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm-c/Error.h":                                "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm-c/Visibility.h":                           "llvm/lib/CodeGen/CodeGenPrepare.cpp",

    # DTLTO drags in Any/LTO/FormatVariadic.
    "llvm/include/llvm/ADT/Any.h":                                "llvm/lib/DTLTO/DTLTO.cpp",
    "llvm/include/llvm/LTO/Config.h":                             "llvm/lib/DTLTO/DTLTO.cpp",
    "llvm/include/llvm/Support/FormatVariadicDetails.h":          "llvm/lib/DTLTO/DTLTO.cpp",

    # Orc debugging / executor-side helpers.
    "llvm/include/llvm/DebugInfo/DWARF/LowLevel/DWARFDataExtractorSimple.h":
        "llvm/lib/ExecutionEngine/Orc/Debugging/DebugInfoSupport.cpp",
    "llvm/include/llvm/ExecutionEngine/Orc/MaterializationUnit.h":
        "llvm/lib/ExecutionEngine/Orc/Debugging/DebugInfoSupport.cpp",
    "llvm/include/llvm/ExecutionEngine/Orc/TargetProcess/ExecutorBootstrapService.h":
        "llvm/lib/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.cpp",
    "llvm/include/llvm-c/blake3.h":
        "llvm/lib/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.cpp",
    "llvm/include/llvm/CodeGen/AsmPrinterHandler.h":
        "llvm/lib/CodeGen/AsmPrinter/DebugHandlerBase.cpp",
}


# Include subdir -> lib subdir remap for cases where the names diverge.
# ADT has no `llvm/lib/ADT/`; its few non-inline implementations live under
# `llvm/lib/Support/`.
INCLUDE_TO_LIB_SUBDIR = {
    "ADT": "Support",
}


# Categories must match `kLlvmCategories` in `llvm/utils/idt/idt.cc`. We
# duplicate the table here only to batch headers per category before
# invoking idt; idt itself derives the export macro / include header / system
# prefixes from the same path-fragment match. Order matters: more specific
# fragments first.
CATEGORIES = [
    ("Demangle", "llvm/Demangle/"),
    ("LLVM-C",   "llvm-c/"),
    ("LLVM",     "llvm/"),
]


def categorize_header(path: str) -> Optional[str]:
    """Return the category name (LLVM/LLVM-C/Demangle) for a header, or None
    if the path doesn't fall under any known category."""
    for name, fragment in CATEGORIES:
        if fragment in path:
            return name
    return None


def find_source_for_header(header: str) -> Optional[str]:
    """Return a source file (relative path) that includes the given header,
    or None if no good match.

    LLVM doesn't list headers in compile_commands.json, so idt needs a
    source TU to parse. With the `hasBody` / out-of-line distinction inside
    idt, we can pick the conventional `Foo.cpp` partner without losing
    coverage of `Foo.h`'s own symbols.

    Resolution order:
    1. Manual override in HEADER_SOURCE_OVERRIDES.
    2. Direct mapping: llvm/include/llvm/Foo/Bar.h -> llvm/lib/Foo/Bar.cpp,
       with INCLUDE_TO_LIB_SUBDIR applied (e.g. ADT -> Support).
    3. Same-subdirectory grep: any .cpp under llvm/lib/Foo/ that
       `#include`s the header. Constrained to the matching subdirectory
       so we don't accidentally pick a target-specific source as a
       generic fallback.
    4. Bail (returns None; the header is silently skipped).
    """
    if header in HEADER_SOURCE_OVERRIDES:
        return HEADER_SOURCE_OVERRIDES[header]

    if not header.startswith("llvm/include/llvm/"):
        return None

    sub = header[len("llvm/include/llvm/"):]
    first_part, _, rest = sub.partition("/")
    remap = INCLUDE_TO_LIB_SUBDIR.get(first_part)
    sub_remapped = f"{remap}/{rest}" if remap and rest else sub
    for ext in (".cpp", ".cc"):
        candidate = Path("llvm/lib") / Path(sub_remapped).with_suffix(ext)
        if candidate.exists():
            return str(candidate)

    lib_subdir = f"llvm/lib/{INCLUDE_TO_LIB_SUBDIR.get(first_part, first_part)}"
    if not Path(lib_subdir).is_dir():
        return None

    inc = header.removeprefix("llvm/include/")
    try:
        result = subprocess.run(
            ["git", "grep", "-l", f'#include "{inc}"', "--", lib_subdir],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
        if line.endswith((".cpp", ".cc")):
            return line
    return None


class IdsCheckArgs:
    start_rev: str = ""
    end_rev: str = ""
    changed_files: List[str] = []
    idt_path: str = ""
    compile_commands: str = ""
    repo: str = ""
    token: str = ""
    issue_number: int = 0
    verbose: bool = True

    def __init__(self, args: argparse.Namespace) -> None:
        self.start_rev = args.start_rev
        self.end_rev = args.end_rev
        self.changed_files = args.changed_files
        self.idt_path = args.idt_path
        self.compile_commands = args.compile_commands
        self.repo = getattr(args, "repo", "")
        self.token = getattr(args, "token", "")
        self.issue_number = getattr(args, "issue_number", 0)
        self.verbose = getattr(args, "verbose", True)


class IdsChecker:
    """Thin orchestrator around the idt binary."""

    COMMENT_TAG = "<!--LLVM IDS CHECK COMMENT-->"
    name = "ids-check"
    friendly_name = "LLVM ABI annotation checker"
    comment: dict = {}

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG

    @property
    def instructions(self) -> str:
        return (
            "Build idt under llvm/utils/idt/, then for each changed header:\n"
            "    idt --header <header> -p build/compile_commands.json \\\n"
            "        --apply-fixits --inplace <matching-source.cpp>"
        )

    def pr_comment_text_for_diff(self, diff: str) -> str:
        return f"""
:warning: {self.friendly_name}, {self.name} found issues in your code. :warning:

<details>
<summary>
You can test this locally with the following command:
</summary>

``````````bash
{self.instructions}
``````````

</details>

<details>
<summary>
View the diff from {self.name} here.
</summary>

``````````diff
{diff}
``````````

</details>
"""

    def update_pr(
        self, comment_text: str, args: IdsCheckArgs, create_new: bool
    ) -> None:
        import github

        repo = github.Github(auth=github.Auth.Token(args.token)).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        comment_text = self.comment_tag + "\n\n" + comment_text

        existing_comment = None
        for comment in pr.as_issue().get_comments():
            if self.comment_tag in comment.body:
                existing_comment = comment
                break

        if existing_comment:
            self.comment = {"body": comment_text, "id": existing_comment.id}
        elif create_new:
            self.comment = {"body": comment_text}

    def run_idt_for_category(
        self, category: str, header_source_pairs: List[tuple],
        args: IdsCheckArgs, idt_path: str, compile_commands: str,
    ) -> None:
        """Invoke idt once for all headers in a category. idt is told the full
        list of headers via repeated --header, and the full list of unique
        source TUs as positional arguments. idt scopes its diagnostics to the
        listed headers and applies fix-its in place."""
        if not header_source_pairs:
            return

        sources = sorted({s for _, s in header_source_pairs})
        cmd = [idt_path, "-p", compile_commands, "--apply-fixits", "--inplace"]
        for h, _ in header_source_pairs:
            cmd += [f"--header={h}"]
        cmd += sources

        if args.verbose:
            print(
                f"Running idt on {len(header_source_pairs)} {category} header(s) "
                f"via {len(sources)} source TU(s)...",
                file=sys.stderr,
            )

        subprocess.run(cmd)

    def get_changed_files(self, args: IdsCheckArgs) -> List[str]:
        if args.changed_files:
            return args.changed_files

        cmd = ["git", "diff", "--name-only", args.start_rev, args.end_rev]
        if args.verbose:
            print(f"Running: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print("Error: Failed to get changed files", file=sys.stderr)
            sys.stderr.write(proc.stderr.decode("utf-8"))
            return []
        files = proc.stdout.decode("utf-8").strip().split("\n")
        return [f for f in files if f]

    def check_for_diff(self) -> Optional[str]:
        proc = subprocess.run(["git", "diff"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        diff = proc.stdout.decode("utf-8")
        return diff if diff else None

    def run(self, args: IdsCheckArgs) -> int:
        idt_path = args.idt_path or os.environ.get("IDT_PATH")
        if not idt_path:
            print("Error: idt path not specified. Use --idt-path or set IDT_PATH.",
                  file=sys.stderr)
            return 1
        if not os.path.exists(idt_path):
            print(f"Error: idt tool not found at {idt_path}", file=sys.stderr)
            return 1

        compile_commands = args.compile_commands or os.environ.get(
            "COMPILE_COMMANDS_PATH"
        )
        if not compile_commands:
            print("Error: --compile-commands not specified.", file=sys.stderr)
            return 1
        if not os.path.exists(compile_commands):
            print(f"Error: compile_commands.json not found at {compile_commands}",
                  file=sys.stderr)
            return 1

        changed_files = self.get_changed_files(args)
        if not changed_files:
            if args.verbose:
                print("No files changed, skipping ids check", file=sys.stderr)
            return 0

        # Filter to relevant headers and group by category.
        per_category = defaultdict(list)
        for path in changed_files:
            if not path.endswith(".h"):
                continue
            if not path.startswith(("llvm/include/llvm/", "llvm/include/llvm-c/")):
                continue
            if any(path.startswith(p) for p in SKIP_HEADERS):
                continue
            category = categorize_header(path)
            if category is None:
                continue
            source = find_source_for_header(path)
            if source is None:
                if args.verbose:
                    print(f"  Skipping {path}: no matching source",
                          file=sys.stderr)
                continue
            per_category[category].append((path, source))

        if not per_category:
            if args.verbose:
                print("No relevant header files changed, skipping ids check",
                      file=sys.stderr)
            return 0

        for category, pairs in per_category.items():
            self.run_idt_for_category(
                category, pairs, args, idt_path, compile_commands
            )

        diff = self.check_for_diff()
        should_update_gh = args.token is not None and args.repo is not None

        if diff:
            if should_update_gh:
                self.update_pr(self.pr_comment_text_for_diff(diff),
                               args, create_new=True)
            else:
                print("\nError: idt found missing LLVM_ABI annotations",
                      file=sys.stderr)
                print("Apply the following diff to fix the LLVM_ABI annotations:\n",
                      file=sys.stderr)
                print(diff)
            return 1

        if should_update_gh:
            self.update_pr(
                ":white_check_mark: With the latest revision "
                f"this PR passed the {self.friendly_name}.",
                args, create_new=False,
            )
        if args.verbose:
            print("All files pass ids check", file=sys.stderr)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check LLVM ABI annotations in header files"
    )
    parser.add_argument("--token", type=str, help="GitHub authentication token")
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", "llvm/llvm-project"),
        help="GitHub repository <owner>/<repo>",
    )
    parser.add_argument("--issue-number", type=int, help="GitHub issue/PR number")
    parser.add_argument("--start-rev", type=str, required=True,
                        help="Compute changes from this revision")
    parser.add_argument("--end-rev", type=str, required=True,
                        help="Compute changes to this revision")
    parser.add_argument("--changed-files", type=str,
                        help="Comma separated list of files that have been changed")
    parser.add_argument("--idt-path", type=str,
                        help="Path to the idt executable (or set IDT_PATH)")
    parser.add_argument("--compile-commands", type=str,
                        help="Path to compile_commands.json (or set COMPILE_COMMANDS_PATH)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Enable verbose output")

    parsed_args = parser.parse_args()

    if parsed_args.changed_files:
        parsed_args.changed_files = [
            f.strip() for f in parsed_args.changed_files.split(",") if f.strip()
        ]
    else:
        parsed_args.changed_files = []

    args = IdsCheckArgs(parsed_args)
    checker = IdsChecker()
    exit_code = checker.run(args)

    if checker.comment:
        with open("comments", "w") as f:
            import json
            json.dump([checker.comment], f)

    sys.exit(exit_code)
