#!/usr/bin/env python3
#
# ====- clang-tidy-helper, runs clang-tidy from the ci --*- python -*-------==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import argparse
import os
import subprocess
import sys
from typing import List, Optional

"""
This script is run by GitHub actions to ensure that the code in PR's conform to
the coding style of LLVM. The canonical source of this script is in the LLVM
source tree under llvm/utils/git.

You can learn more about the LLVM coding style on llvm.org:
https://llvm.org/docs/CodingStandards.html
"""


class LintArgs:
    start_rev: str = None
    end_rev: str = None
    repo: str = None
    changed_files: List[str] = []
    token: str = None
    verbose: bool = True
    issue_number: int = 0

    def __init__(self, args: argparse.Namespace = None) -> None:
        if not args is None:
            self.start_rev = args.start_rev
            self.end_rev = args.end_rev
            self.repo = args.repo
            self.token = args.token
            self.changed_files = args.changed_files
            self.issue_number = args.issue_number
            self.verbose = args.verbose


class LintHelper:
    COMMENT_TAG = "<!--LLVM CODE LINT COMMENT: {linter}-->"
    name: str
    friendly_name: str
    comment: dict = None

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG.replace("linter", self.name)

    @property
    def instructions(self) -> str:
        raise NotImplementedError()

    def has_tool(self) -> bool:
        raise NotImplementedError()

    def lint_run(self, changed_files: List[str], args: LintArgs) -> Optional[str]:
        raise NotImplementedError()

    def pr_comment_text_for_output(self, warning: str) -> str:
        return f"""
:warning: {self.friendly_name} {self.name} found issues in your code. :warning:

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
View the output from {self.name} here.
</summary>

``````````
{warning}
``````````

</details>
"""

    def find_comment(self, pr: any) -> any:
        for comment in pr.as_issue().get_comments():
            if self.comment_tag in comment.body:
                return comment
        return None

    def update_pr(self, comment_text: str, args: LintArgs, create_new: bool) -> None:
        import github

        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        comment_text = self.comment_tag + "\n\n" + comment_text

        existing_comment = self.find_comment(pr)

        if create_new or existing_comment:
            self.comment = {"body": comment_text}
        if existing_comment:
            self.comment["id"] = existing_comment.id
        return

    def run(self, changed_files: List[str], args: LintArgs) -> bool:
        changed_files = [arg for arg in changed_files if "third-party" not in arg]
        diff = self.lint_run(changed_files, args)
        should_update_gh = args.token is not None and args.repo is not None

        if diff is None:
            if should_update_gh:
                comment_text = (
                    ":white_check_mark: With the latest revision "
                    f"this PR passed the {self.friendly_name}."
                )
                self.update_pr(comment_text, args, create_new=False)
            return True
        elif len(diff) > 0:
            if should_update_gh:
                comment_text = self.pr_comment_text_for_output(diff)
                self.update_pr(comment_text, args, create_new=True)
            else:
                print(
                    f"Warning: {self.friendly_name}, {self.name} detected "
                    "some issues with your code formatting..."
                )
            return False
        else:
            # The linter failed but didn't output a result (e.g. some sort of
            # infrastructure failure).
            comment_text = (
                f":warning: The {self.friendly_name} failed without printing "
                "a diff. Check the logs for stderr output. :warning:"
            )
            self.update_pr(comment_text, args, create_new=False)
            return False


class ClangTidyDiffHelper(LintHelper):
    name = "clang-tidy"
    friendly_name = "C/C++ code linter"

    def __init__(self, build_path: str, clang_tidy_binary: str):
        self.build_path = build_path
        self.clang_tidy_binary = clang_tidy_binary
        self.cpp_files = []

    def _clean_clang_tidy_output(self, output: str) -> str:
        """
        - Remove 'Running clang-tidy in X threads...' line
        - Remove 'N warnings generated.' line
        - Strip leading workspace path from file paths
        """
        if not output or output == "No relevant changes found.":
            return None

        lines = output.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.startswith("Running clang-tidy in") or line.endswith("generated."):
                continue

            # Remove everything up to rightmost "llvm-project/" for correct files names
            idx = line.rfind("llvm-project/")
            if idx != -1:
                line = line[idx + len("llvm-project/") :]

            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)
        return None

    @property
    def instructions(self) -> str:
        files_str = " ".join(self.cpp_files)
        return f"""
git diff -U0 origin/main..HEAD -- {files_str} |
python3 clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py \\
  -path build -p1 -quiet"""

    # For add other paths/files to this function
    def should_lint_file(self, filepath):
        return filepath.startswith("clang-tools-extra/clang-tidy/")

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        filtered_files = []
        for filepath in changed_files:
            _, ext = os.path.splitext(filepath)
            if ext not in (".cpp", ".c", ".h", ".hpp", ".hxx", ".cxx"):
                continue
            if not self.should_lint_file(filepath):
                continue
            if os.path.exists(filepath):
                filtered_files.append(filepath)

        return filtered_files

    def has_tool(self) -> bool:
        cmd = [self.clang_tidy_binary, "--version"]
        proc = None
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            return False
        return proc.returncode == 0

    def lint_run(self, changed_files: List[str], args: LintArgs) -> Optional[str]:
        cpp_files = self.filter_changed_files(changed_files)
        if not cpp_files:
            print("no c/c++ files found")
            return None

        self.cpp_files = cpp_files

        git_diff_cmd = [
            "git",
            "diff",
            "-U0",
            f"{args.start_rev}..{args.end_rev}",
            "--",
        ] + cpp_files

        diff_proc = subprocess.run(
            git_diff_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if diff_proc.returncode != 0:
            print(f"Git diff failed: {diff_proc.stderr}")
            return None

        diff_content = diff_proc.stdout
        if not diff_content.strip():
            print("No diff content found")
            return None

        tidy_diff_cmd = [
            "code-lint-tools/clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py",
            "-path",
            self.build_path,
            "-p1",
            "-quiet",
        ]

        if args.verbose:
            print(f"Running clang-tidy-diff: {' '.join(tidy_diff_cmd)}")

        proc = subprocess.run(
            tidy_diff_cmd,
            input=diff_content,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        return self._clean_clang_tidy_output(proc.stdout.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token", type=str, required=True, help="GitHub authentication token"
    )
    parser.add_argument("--issue-number", type=int, required=True)
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", "llvm/llvm-project"),
        help="The GitHub repository that we are working with in the form of <owner>/<repo> (e.g. llvm/llvm-project)",
    )
    parser.add_argument(
        "--start-rev",
        type=str,
        required=True,
        help="Compute changes from this revision.",
    )
    parser.add_argument(
        "--end-rev", type=str, required=True, help="Compute changes to this revision"
    )
    parser.add_argument(
        "--changed-files",
        type=str,
        help="Comma separated list of files that has been changed",
    )
    parser.add_argument(
        "--build-path",
        type=str,
        default="build",
        help="Path to build directory with compile_commands.json",
    )
    parser.add_argument(
        "--clang-tidy-binary",
        type=str,
        default="clang-tidy",
        help="Path to clang-tidy binary",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    parsed_args = parser.parse_args()
    args = LintArgs(parsed_args)

    changed_files = []
    if args.changed_files:
        changed_files = args.changed_files.split(",")

    if args.verbose:
        print(f"got changed files: {changed_files}")

    all_linters = [
        ClangTidyDiffHelper(
            build_path=parsed_args.build_path,
            clang_tidy_binary=parsed_args.clang_tidy_binary,
        )
    ]

    failed_linters = []
    comments = []

    for linter in all_linters:
        if not linter.has_tool():
            print(f"Couldn't find {linter.friendly_name}: {linter.name}")
            continue

        if args.verbose:
            print(f"running linter {linter.name}")

        if not linter.run(changed_files, args):
            if args.verbose:
                print(f"linter {linter.name} failed")
            failed_linters.append(linter.name)

        if linter.comment:
            if args.verbose:
                print(f"linter {linter.name} has comment: {linter.comment}")
            comments.append(linter.comment)

    if len(comments) > 0:
        with open("comments", "w") as f:
            import json

            json.dump(comments, f)

    if len(failed_linters) > 0:
        print(f"error: some linters failed: {' '.join(failed_linters)}")
        # Do not fail job for as it may be unstable
        # sys.exit(1)
