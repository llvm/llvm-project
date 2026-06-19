#!/usr/bin/env python3
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import json
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
output filtering, the `hasBody` / out-of-line distinction) lives in `idt`
itself (the standalone `compnerd/ids` tool, built separately and passed
via `--idt-path`). This script is a thin orchestrator: it picks which
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
    # These are not LLVM component libraries.
    "llvm/include/llvm/HTTP/",
    "llvm/include/llvm/Debuginfod/",
    # These are meant to be included via `Pass.h`, not directly.
    "llvm/include/llvm/PassAnalysisSupport.h",
    "llvm/include/llvm/PassSupport.h",
    # Test-only headers.
    "llvm/include/llvm/Testing/",
    # OProfile JIT bridge, disabled by default.
    "llvm/include/llvm/ExecutionEngine/OProfileWrapper.h",
    "llvm/include/llvm/ExecutionEngine/RuntimeDyld.h",
    # PDB DIA requires Windows ATL (atlbase.h), unbuildable on Linux runners.
    "llvm/include/llvm/DebugInfo/PDB/DIA/",
    "llvm/include/llvm/DebugInfo/PDB/ConcreteSymbolEnumerator.h",
    # No in-tree non-tools/non-target source #includes these headers, so we
    # can't run idt on a TU that pulls them in.
    "llvm/include/llvm/ExecutionEngine/Interpreter.h",
    "llvm/include/llvm/Support/DebugLog.h",
    "llvm/include/llvm/Support/TargetSelect.h",
    # zOS-specific shims.
    "llvm/include/llvm/Support/AutoConvert.h",
    # `class LLVM_ABI DagInit` (and similar in this file) inherit from
    # `TrailingObjects<DagInit, T1, T2>` with multiple trailing types.
    # MSVC `__declspec(dllexport)` on the class forces instantiation of
    # all members including `getTrailingObjects()` (no-arg), which has a
    # `static_assert(sizeof...(TrailingTys) == 1, ...)` in its body. The
    # build then fails with C2338. Until `TrailingObjects.h` is updated
    # to gate the no-arg overload via SFINAE / requires-clause, skip the
    # whole header.
    "llvm/include/llvm/TableGen/Record.h",
    # Pimpl classes annotated `LLVM_ABI` that hold `std::unique_ptr<T>`
    # of a forward-declared T defined only in the matching .cpp. MSVC's
    # class-level `__declspec(dllexport)` forces the virtual destructor
    # to be emitted inline as part of the exported vtable, which requires
    # T to be complete in every consuming TU. Pulling T's definition into
    # the public header would leak the implementation, so the right move
    # is to keep these out of the bulk pass. Long-term: teach idt to
    # detect this pattern (unique_ptr<forward-decl> + virtual destructor)
    # and annotate individual methods instead of the class.
    "llvm/include/llvm/MC/MCWinCOFFObjectWriter.h",
    # Wraps the Visual Studio "Setup Configuration" COM API. The
    # declarations rely on `EXTERN_C` / `MAXUINT` from <windows.h>, which
    # idt's standalone parse doesn't pull in. Same long-term fix shape as
    # `AutoConvert.h` (zOS): guard the annotations in-source, or include
    # the prerequisite header. Skip for now.
    "llvm/include/llvm/WindowsDriver/MSVCSetupApi.h",
]


# Manual header -> source mappings for headers that don't fit the conventional
# `llvm/include/llvm/<Subsystem>/<Bar>.h` -> `llvm/lib/<Subsystem>/<Bar>.cpp`
# layout. Used when neither the direct mapping nor the same-subdirectory
# grep fallback finds a workable source.
HEADER_SOURCE_OVERRIDES = {
    # LLVM-C headers
    "llvm/include/llvm-c/Analysis.h": "llvm/lib/Analysis/Analysis.cpp",
    "llvm/include/llvm-c/BitReader.h": "llvm/lib/Bitcode/Reader/BitReader.cpp",
    "llvm/include/llvm-c/BitWriter.h": "llvm/lib/Bitcode/Writer/BitWriter.cpp",
    "llvm/include/llvm-c/Comdat.h": "llvm/lib/IR/Comdat.cpp",
    "llvm/include/llvm-c/Core.h": "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm-c/DebugInfo.h": "llvm/lib/IR/DebugInfo.cpp",
    "llvm/include/llvm-c/Disassembler.h": "llvm/lib/MC/MCDisassembler/Disassembler.cpp",
    "llvm/include/llvm-c/ErrorHandling.h": "llvm/lib/Support/ErrorHandling.cpp",
    "llvm/include/llvm-c/ExecutionEngine.h": "llvm/lib/ExecutionEngine/ExecutionEngineBindings.cpp",
    "llvm/include/llvm-c/IRReader.h": "llvm/lib/IRReader/IRReader.cpp",
    "llvm/include/llvm-c/LLJIT.h": "llvm/lib/ExecutionEngine/Orc/OrcV2CBindings.cpp",
    "llvm/include/llvm-c/LLJITUtils.h": "llvm/lib/ExecutionEngine/Orc/Debugging/LLJITUtilsCBindings.cpp",
    "llvm/include/llvm-c/Linker.h": "llvm/lib/Linker/LinkModules.cpp",
    "llvm/include/llvm-c/Object.h": "llvm/lib/Object/Object.cpp",
    "llvm/include/llvm-c/Orc.h": "llvm/lib/ExecutionEngine/Orc/OrcV2CBindings.cpp",
    "llvm/include/llvm-c/OrcEE.h": "llvm/lib/ExecutionEngine/Orc/OrcV2CBindings.cpp",
    "llvm/include/llvm-c/Remarks.h": "llvm/lib/Remarks/RemarkParser.cpp",
    "llvm/include/llvm-c/Support.h": "llvm/lib/Support/CommandLine.cpp",
    "llvm/include/llvm-c/Target.h": "llvm/lib/Target/Target.cpp",
    "llvm/include/llvm-c/TargetMachine.h": "llvm/lib/Target/TargetMachineC.cpp",
    "llvm/include/llvm-c/Transforms/PassBuilder.h": "llvm/lib/Passes/PassBuilderBindings.cpp",
    "llvm/include/llvm-c/Types.h": "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm-c/lto.h": "llvm/tools/lto/lto.cpp",
    # Top-level llvm/ headers
    "llvm/include/llvm/Pass.h": "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/PassAnalysisSupport.h": "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/PassRegistry.h": "llvm/lib/IR/PassRegistry.cpp",
    "llvm/include/llvm/PassSupport.h": "llvm/lib/IR/Pass.cpp",
    "llvm/include/llvm/InitializePasses.h": "llvm/lib/Analysis/Analysis.cpp",
    "llvm/include/llvm/LinkAllIR.h": "llvm/lib/IR/Core.cpp",
    "llvm/include/llvm/LinkAllPasses.h": "llvm/lib/IR/Pass.cpp",
    # Headers under llvm/include/llvm/Target/ whose same-subdir grep fallback
    # picks per-target sources (e.g. X86, AArch64). Redirect to lib/CodeGen/.
    "llvm/include/llvm/Target/CGPassBuilderOption.h": "llvm/lib/CodeGen/TargetPassConfig.cpp",
    "llvm/include/llvm/Target/TargetOptions.h": "llvm/lib/CodeGen/AsmPrinter/AsmPrinter.cpp",
    # Headers pulled in transitively by a small set of "umbrella" sources.
    "llvm/include/llvm/ADT/ilist_node_base.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/Analysis/SimplifyQuery.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/CodeGenTypes/MachineValueType.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/Analysis.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/ConstantFolder.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/FMF.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/GenericFloatingPointPredicateUtils.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/IR/IRBuilderFolder.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm/Support/Recycler.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm-c/Error.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    "llvm/include/llvm-c/Visibility.h": "llvm/lib/CodeGen/CodeGenPrepare.cpp",
    # DTLTO drags in Any/LTO/FormatVariadic.
    "llvm/include/llvm/ADT/Any.h": "llvm/lib/DTLTO/DTLTO.cpp",
    "llvm/include/llvm/LTO/Config.h": "llvm/lib/DTLTO/DTLTO.cpp",
    "llvm/include/llvm/Support/FormatVariadicDetails.h": "llvm/lib/DTLTO/DTLTO.cpp",
    # Orc debugging / executor-side helpers.
    "llvm/include/llvm/DebugInfo/DWARF/LowLevel/DWARFDataExtractorSimple.h": "llvm/lib/ExecutionEngine/Orc/Debugging/DebugInfoSupport.cpp",
    "llvm/include/llvm/ExecutionEngine/Orc/MaterializationUnit.h": "llvm/lib/ExecutionEngine/Orc/Debugging/DebugInfoSupport.cpp",
    "llvm/include/llvm/ExecutionEngine/Orc/TargetProcess/ExecutorBootstrapService.h": "llvm/lib/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.cpp",
    "llvm/include/llvm-c/blake3.h": "llvm/lib/ExecutionEngine/Orc/TargetProcess/ExecutorSharedMemoryMapperService.cpp",
    "llvm/include/llvm/CodeGen/AsmPrinterHandler.h": "llvm/lib/CodeGen/AsmPrinter/DebugHandlerBase.cpp",
}


# Include subdir -> lib subdir remap for cases where the names diverge.
# ADT has no `llvm/lib/ADT/`; its few non-inline implementations live under
# `llvm/lib/Support/`.
INCLUDE_TO_LIB_SUBDIR = {
    "ADT": "Support",
}


# Categories: each header path-fragment maps to (category-name, export-macro).
# Order matters: more specific fragments first (Demangle before generic LLVM).
CATEGORIES = [
    ("Demangle", "llvm/Demangle/", "DEMANGLE_ABI"),
    ("LLVM-C", "llvm-c/", "LLVM_C_ABI"),
    ("LLVM", "llvm/", "LLVM_ABI"),
]

EXPORT_MACRO = {name: macro for name, _fragment, macro in CATEGORIES}

# Predefines applied to every idt invocation (via `--extra-arg`).
#
# Why these are required: idt detects an existing annotation by inspecting
# clang's parsed attributes (`DLLExportAttr` / `DLLImportAttr` /
# `VisibilityAttr`). In a static LLVM build, the donor source's compile
# command carries `-DLLVM_BUILD_STATIC`, which collapses the macro guards
# in `Compiler.h` / `DemangleConfig.h` / `llvm-c/Visibility.h` so that
# `LLVM_ABI` / `DEMANGLE_ABI` / `LLVM_C_ABI` all expand to *empty*. The
# parser then sees no attribute on already-annotated declarations, and
# idt happily prepends another copy of the macro on top.
#
# The fix is to make those macros expand to a real attribute regardless
# of the donor's static-vs-dylib choice:
#  - `-ULLVM_BUILD_STATIC` removes the kill-switch.
#  - `-DLLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS` opens the LLVM_ABI /
#    DEMANGLE_ABI gates.
#  - `-DLLVM_ENABLE_LLVM_C_EXPORT_ANNOTATIONS` opens the LLVM_C_ABI gate.
#  - `-DLLVM_EXPORTS` makes the cascade pick `dllexport` over `dllimport`
#    (either would suffice for detection, dllexport is the symmetric
#    choice for a "we're inside the library" parse).
EXTRA_ARGS_COMMON = [
    "-ULLVM_BUILD_STATIC",
    "-DLLVM_ENABLE_LLVM_EXPORT_ANNOTATIONS",
    "-DLLVM_ENABLE_LLVM_C_EXPORT_ANNOTATIONS",
    "-DLLVM_EXPORTS",
]


def categorize_header(path: str) -> Optional[str]:
    """Return the category name (LLVM/LLVM-C/Demangle) for a header, or None
    if the path doesn't fall under any known category."""
    for name, fragment, _macro in CATEGORIES:
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

    # 1. Manual override for special cases.
    if header in HEADER_SOURCE_OVERRIDES:
        return HEADER_SOURCE_OVERRIDES[header]

    if not header.startswith("llvm/include/llvm/"):
        return None

    # 2. Direct mapping.
    sub = header[len("llvm/include/llvm/") :]
    first_part, _, rest = sub.partition("/")
    remap = INCLUDE_TO_LIB_SUBDIR.get(first_part)
    sub_remapped = f"{remap}/{rest}" if remap and rest else sub
    for ext in (".cpp", ".cc"):
        candidate = Path("llvm/lib") / Path(sub_remapped).with_suffix(ext)
        if candidate.exists():
            return str(candidate)

    # 3. Same-subdirectory grep fallback.
    lib_subdir = f"llvm/lib/{INCLUDE_TO_LIB_SUBDIR.get(first_part, first_part)}"
    if not Path(lib_subdir).is_dir():
        return None

    inc = header.removeprefix("llvm/include/")
    try:
        result = subprocess.run(
            ["git", "grep", "-l", f'#include "{inc}"', "--", lib_subdir],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0 or not result.stdout:
        return None
    for line in result.stdout.splitlines():
        if line.endswith((".cpp", ".cc")):
            return line

    # 4. Bail.
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
    verbose: bool = False

    def __init__(self, args: argparse.Namespace) -> None:
        self.start_rev = args.start_rev
        self.end_rev = args.end_rev
        self.changed_files = args.changed_files
        self.idt_path = args.idt_path
        self.compile_commands = args.compile_commands
        self.repo = getattr(args, "repo", "")
        self.token = getattr(args, "token", "")
        self.issue_number = getattr(args, "issue_number", 0)
        self.verbose = getattr(args, "verbose", False)


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
            "Build idt from compnerd/ids, then for each changed header:\n"
            "    idt -p build/ --main-file <matching-source.cpp> \\\n"
            "        --apply-fixits --inplace <header>"
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
        self,
        category: str,
        header_source_pairs: List[tuple],
        args: IdsCheckArgs,
        idt_path: str,
        compile_commands: str,
    ) -> None:
        """Invoke idt once per header, using the matching source as the
        flag donor via idt's `--main-file` option. idt looks the donor up
        in compile_commands.json, swaps its input argument for the
        positional header, and forces `-x c++-header` — so the header is
        parsed standalone as the translation unit.

        Per-header subprocess invocation also sidesteps idt's known
        cross-TU state issues (DiagnosticsEngine instance lifetime in
        batched runs).
        """
        if not header_source_pairs:
            return

        if args.verbose:
            print(
                f"Running idt on {len(header_source_pairs)} {category} file(s)...",
                file=sys.stderr,
            )

        def fwd(p: str) -> str:
            # Forward slashes so paths survive cl::TokenizeGNUCommandLine.
            return p.replace("\\", "/")

        # Resolve `source` to the absolute path stored in
        # compile_commands.json so idt's CompilationDatabase lookup is
        # exact rather than relying on suffix-matching heuristics.
        try:
            with open(compile_commands, "r", encoding="utf-8") as f:
                cdb_entries = json.load(f)
        except Exception as e:
            print(f"  ERROR: failed to read {compile_commands}: {e}", file=sys.stderr)
            return
        cdb_paths = [fwd(e["file"]) for e in cdb_entries]

        cdb_dir = os.path.dirname(compile_commands) or "."

        for i, (header, source) in enumerate(header_source_pairs, start=1):
            if args.verbose:
                print(
                    f"  [{i}/{len(header_source_pairs)}] {header} "
                    f"(flags from {source})",
                    file=sys.stderr,
                )

            source_norm = fwd(source)
            donor_path = next((p for p in cdb_paths if p.endswith(source_norm)), None)
            if donor_path is None:
                print(
                    f"  WARN: no compile_commands entry matching {source}; "
                    f"skipping {header}.",
                    file=sys.stderr,
                )
                continue

            cmd = [
                idt_path,
                "-p",
                fwd(cdb_dir),
                f"--main-file={donor_path}",
                f"--export-macro={EXPORT_MACRO[category]}",
                "--apply-fixits",
                "--inplace",
            ]
            for extra in EXTRA_ARGS_COMMON:
                cmd.append(f"--extra-arg={extra}")
            cmd.append(fwd(header))
            proc = subprocess.run(cmd)
            if proc.returncode != 0 and proc.returncode != 1:
                print(
                    f"  WARN: idt exited with rc={proc.returncode} on "
                    f"{header}; continuing.",
                    file=sys.stderr,
                )

    def get_changed_files(self, args: IdsCheckArgs) -> List[str]:
        """Get list of changed files between revisions."""
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
        """Check if there are any uncommitted changes after running idt."""
        cmd = ["git", "diff"]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        diff = proc.stdout.decode("utf-8")
        if diff:
            return diff
        return None

    def run(self, args: IdsCheckArgs) -> int:
        """Main entry point for running ids check."""
        # Resolve idt path: prefer command line arg, then env var
        idt_path = args.idt_path or os.environ.get("IDT_PATH")
        if not idt_path:
            print(
                "Error: idt path not specified. Use --idt-path argument or set IDT_PATH environment variable",
                file=sys.stderr,
            )
            return 1

        if not os.path.exists(idt_path):
            print(f"Error: idt tool not found at {idt_path}", file=sys.stderr)
            return 1

        # Resolve compile_commands path: prefer command line arg, then env var
        compile_commands = args.compile_commands or os.environ.get(
            "COMPILE_COMMANDS_PATH"
        )
        if not compile_commands:
            print(
                "Error: compile_commands.json path not specified. Use --compile-commands argument or set COMPILE_COMMANDS_PATH environment variable",
                file=sys.stderr,
            )
            return 1

        if not os.path.exists(compile_commands):
            print(
                f"Error: compile_commands.json not found at {compile_commands}",
                file=sys.stderr,
            )
            return 1

        # Get changed files
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
                    print(f"  Skipping {path}: no matching source", file=sys.stderr)
                continue
            per_category[category].append((path, source))

        if not per_category:
            if args.verbose:
                print(
                    "No relevant header files changed, skipping ids check",
                    file=sys.stderr,
                )
            return 0

        for category, pairs in per_category.items():
            self.run_idt_for_category(category, pairs, args, idt_path, compile_commands)

        # Check for differences
        diff = self.check_for_diff()
        should_update_gh = args.token is not None and args.repo is not None

        if diff:
            if should_update_gh:
                comment_text = self.pr_comment_text_for_diff(diff)
                self.update_pr(comment_text, args, create_new=True)
            else:
                print(
                    "\nError: idt found missing LLVM_ABI annotations", file=sys.stderr
                )
                print(
                    "Apply the following diff to fix the LLVM_ABI annotations:\n",
                    file=sys.stderr,
                )
                print(diff)
            return 1
        else:
            if should_update_gh:
                comment_text = (
                    ":white_check_mark: With the latest revision "
                    f"this PR passed the {self.friendly_name}."
                )
                self.update_pr(comment_text, args, create_new=False)
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
        help="The GitHub repository that we are working with in the form of <owner>/<repo> (e.g. llvm/llvm-project)",
    )
    parser.add_argument("--issue-number", type=int, help="GitHub issue/PR number")
    parser.add_argument(
        "--start-rev",
        type=str,
        required=True,
        help="Compute changes from this revision",
    )
    parser.add_argument(
        "--end-rev",
        type=str,
        required=True,
        help="Compute changes to this revision",
    )
    parser.add_argument(
        "--changed-files",
        type=str,
        help="Comma-separated list of changed files, or `@<path>` to read "
        "the list from a response file (one path per line or "
        "comma-separated). The `@`-prefix follows the standard "
        "compiler/linker convention and is useful on Windows where the "
        "inline comma-separated form can hit command-line length limits.",
    )
    parser.add_argument(
        "--idt-path",
        type=str,
        help="Path to the idt executable (can also be set via IDT_PATH environment variable)",
    )
    parser.add_argument(
        "--compile-commands",
        type=str,
        help="Path to compile_commands.json (can also be set via COMPILE_COMMANDS_PATH environment variable)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parsed_args = parser.parse_args()

    raw = parsed_args.changed_files or ""
    if raw.startswith("@"):
        # Response file: read its contents and treat the inline value as if
        # it had been spelled inline.
        with open(raw[1:], "r", encoding="utf-8") as f:
            raw = f.read()

    # Accept both comma-separated and newline-separated entries so the
    # response-file variant is friendly to write line-by-line.
    parsed_args.changed_files = [
        token.strip()
        for line in raw.splitlines()
        for token in line.split(",")
        if token.strip()
    ]

    args = IdsCheckArgs(parsed_args)
    checker = IdsChecker()
    exit_code = checker.run(args)

    if checker.comment:
        with open("comments", "w") as f:
            import json

            json.dump([checker.comment], f)

    sys.exit(exit_code)
