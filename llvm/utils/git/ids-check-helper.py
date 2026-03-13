#!/usr/bin/env python3
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import argparse
import os
import subprocess
import sys
from typing import List, Optional

"""
This script is run by GitHub actions to ensure that the code in PRs properly
labels LLVM APIs with `LLVM_ABI` so as not to break the LLVM DLL build. It can
also be installed as a pre-commit git hook to check ABI annotations before
submitting. The canonical source of this script is in the LLVM source tree
under llvm/utils/git.

This script uses the idt (Interface Diff Tool) to check for missing LLVM_ABI,
LLVM_C_ABI, and DEMANGLE_ABI annotations in header files.

You can install this script as a git hook by symlinking it to the .git/hooks
directory:

ln -s $(pwd)/llvm/utils/git/ids-check-helper.py .git/hooks/pre-commit

You can control the exact path to idt and compile_commands.json with the
following environment variables: $IDT_PATH and $COMPILE_COMMANDS_PATH.
"""


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
    """
    Checker for LLVM ABI annotations using the idt tool.
    """

    COMMENT_TAG = "<!--LLVM IDS CHECK COMMENT-->"
    name = "ids-check"
    friendly_name = "LLVM ABI annotation checker"
    comment: dict = {}

    # Macro definition used for all export macros
    MACRO_DEFINITION = '__attribute__((visibility("default")))'

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG

    @property
    def instructions(self) -> str:
        # Provide basic usage instructions
        return f"""git diff origin/main HEAD -- 'llvm/include/llvm/**/*.h' 'llvm/include/llvm-c/**/*.h' 'llvm/include/llvm/Demangle/**/*.h'
Then run idt on the changed files with appropriate --export-macro and --include-header flags."""

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

:warning:
The reproduction instructions above might return results for more than one PR
in a stack if you are using a stacked PR workflow. You can limit the results by
changing `origin/main` to the base branch/commit you want to compare against.
:warning:

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

    # Define the file categories and their corresponding configurations
    FILE_CATEGORIES = [
        {
            "name": "LLVM headers",
            "patterns": ["llvm/include/llvm/**/*.h"],
            "excludes": [
                "llvm/include/llvm/Debuginfod/",
                "llvm/include/llvm/Demangle/",
            ],
            "export_macro": "LLVM_ABI",
            "include_header": "llvm/Support/Compiler.h",
        },
        {
            "name": "LLVM-C headers",
            "patterns": ["llvm/include/llvm-c/**/*.h"],
            "excludes": [],
            "export_macro": "LLVM_C_ABI",
            "include_header": "llvm-c/Visibility.h",
        },
        {
            "name": "LLVM Demangle headers",
            "patterns": ["llvm/include/llvm/Demangle/**/*.h"],
            "excludes": [],
            "export_macro": "DEMANGLE_ABI",
            "include_header": "llvm/Demangle/Visibility.h",
        },
    ]

    def filter_files_for_category(
        self, changed_files: List[str], category: dict
    ) -> List[str]:
        """Filter changed files based on category patterns and excludes."""
        filtered = []
        for path in changed_files:
            # Check if file matches any pattern
            matches_pattern = False
            for pattern in category["patterns"]:
                # Simple pattern matching for **/*.h style patterns
                pattern_prefix = pattern.replace("**/*.h", "")
                if path.startswith(pattern_prefix) and path.endswith(".h"):
                    matches_pattern = True
                    break

            if not matches_pattern:
                continue

            # Check if file should be excluded
            excluded = False
            for exclude in category["excludes"]:
                if path.startswith(exclude):
                    excluded = True
                    break

            if not excluded:
                filtered.append(path)

        return filtered

    def run_idt_on_files(
        self,
        files: List[str],
        category: dict,
        args: IdsCheckArgs,
        idt_path: str,
        compile_commands: str,
    ) -> bool:
        """Run idt tool on the given files with category-specific configuration."""
        if not files:
            return True

        if args.verbose:
            print(
                f"Running idt on {len(files)} {category['name']} file(s)...",
                file=sys.stderr,
            )

        for file in files:
            cmd = [
                idt_path,
                "-p",
                compile_commands,
                "--apply-fixits",
                "--inplace",
                f"--export-macro={category['export_macro']}",
                f"--include-header={category['include_header']}",
                f"--extra-arg=-D{category['export_macro']}={self.MACRO_DEFINITION}",
                "--extra-arg=-Wno-macro-redefined",
                file,
            ]

            if args.verbose:
                print(f"Running: {' '.join(cmd)}", file=sys.stderr)

            subprocess.run(cmd)

        return True

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

        # Process each category
        any_processed = False
        for category in self.FILE_CATEGORIES:
            filtered_files = self.filter_files_for_category(changed_files, category)
            if filtered_files:
                any_processed = True
                if not self.run_idt_on_files(
                    filtered_files, category, args, idt_path, compile_commands
                ):
                    return 1

        if not any_processed:
            if args.verbose:
                print(
                    "No relevant header files changed, skipping ids check",
                    file=sys.stderr,
                )
            return 0

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
        help="Comma separated list of files that have been changed",
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
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Enable verbose output"
    )

    parsed_args = parser.parse_args()

    # Parse changed files if provided
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
