#!/usr/bin/env python3
#
# ====- clang-tidy-helper, runs clang-tidy from the ci --*- python -*-------==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#
"""A helper script to run clang-tidy linter in GitHub actions

This script is run by GitHub actions to ensure that the code in PR's conform to
the coding style of LLVM. The canonical source of this script is in the LLVM
source tree under llvm/utils/git.

You can learn more about the LLVM coding style on llvm.org:
https://llvm.org/docs/CodingStandards.html
"""

import argparse
import os
import subprocess
import sys
from typing import List, Optional


class LintArgs:
    start_rev: str = None
    end_rev: str = None
    repo: str = None
    changed_files: List[str] = []
    token: str = None
    verbose: bool = True
    issue_number: int = 0
    build_path: str = "build"
    clang_tidy_binary: str = "clang-tidy"
    doc8_binary: str = "doc8"

    def __init__(self, args: argparse.Namespace = None) -> None:
        if not args is None:
            self.start_rev = args.start_rev
            self.end_rev = args.end_rev
            self.repo = args.repo
            self.token = args.token
            if args.changed_files:
                self.changed_files = args.changed_files.split(",")
            else:
                self.changed_files = []
            self.issue_number = args.issue_number
            self.verbose = args.verbose
            self.build_path = args.build_path
            self.clang_tidy_binary = args.clang_tidy_binary
            self.doc8_binary = args.doc8_binary


class LintHelper:
    COMMENT_TAG = "<!--LLVM CODE LINT COMMENT: {linter}-->"
    name: str
    friendly_name: str
    comment: dict = None

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG.format(linter=self.name)

    @property
    def instructions(self) -> str:
        raise NotImplementedError()

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        raise NotImplementedError()

    def run_linter_tool(
        self, files_to_lint: List[str], args: LintArgs
    ) -> Optional[str]:
        raise NotImplementedError()

    def pr_comment_text_for_diff(
        self, linter_output: str, files_to_lint: List[str], args: LintArgs
    ) -> str:
        instructions = self.instructions(files_to_lint, args)
        return f"""
:warning: {self.friendly_name}, {self.name} found issues in your code. :warning:

<details>
<summary>
You can test this locally with the following command:
</summary>

```bash
{instructions}
```

</details>

<details>
<summary>
View the output from {self.name} here.
</summary>

```
{linter_output}
```

</details>
"""

    def find_comment(self, pr: any) -> any:
        for comment in pr.as_issue().get_comments():
            if self.comment_tag in comment.body:
                return comment
        return None

    def update_pr(self, comment_text: str, args: LintArgs, create_new: bool) -> None:
        import github
        from github import IssueComment, PullRequest

        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        comment_text = self.comment_tag + "\n\n" + comment_text

        existing_comment = self.find_comment(pr)

        if create_new or existing_comment:
            self.comment = {"body": comment_text}
        if existing_comment:
            self.comment["id"] = existing_comment.id


    def run(self, args: LintArgs) -> bool:
        files_to_lint = self.filter_changed_files(args.changed_files)

        is_success = True
        linter_output = None

        if files_to_lint:
            linter_output = self.run_linter_tool(files_to_lint, args)
            if linter_output:
                is_success = False

        should_update_gh = args.token is not None and args.repo is not None

        if is_success:
            if should_update_gh:
                comment_text = (
                    ":white_check_mark: With the latest revision "
                    f"this PR passed the {self.friendly_name}."
                )
                self.update_pr(comment_text, args, create_new=False)
            return True
        else:
            if should_update_gh:
                if linter_output:
                    comment_text = self.pr_comment_text_for_diff(
                        linter_output, files_to_lint, args
                    )
                    self.update_pr(comment_text, args, create_new=True)
                else:
                    comment_text = (
                        f":warning: The {self.friendly_name} failed without printing "
                        "an output. Check the logs for output. :warning:"
                    )
                    self.update_pr(comment_text, args, create_new=False)
            else:
                if linter_output:
                    print(
                        f"Warning: {self.friendly_name}, {self.name} detected "
                        "some issues with your code..."
                    )
                    print(linter_output)
                else:
                    print(f"Warning: {self.friendly_name}, {self.name} failed to run.")
            return False


class ClangTidyLintHelper(LintHelper):
    name = "clang-tidy"
    friendly_name = "C/C++ code linter"

    def instructions(self, cpp_files: List[str], args: LintArgs) -> str:
        files_str = " ".join(cpp_files)
        return f"""
git diff -U0 origin/main...HEAD -- {files_str} |
python3 clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py \\
  -path {args.build_path} -p1 -quiet"""

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        clang_tidy_changed_files = [
            arg for arg in changed_files if "third-party" not in arg
        ]

        filtered_files = []
        for filepath in clang_tidy_changed_files:
            _, ext = os.path.splitext(filepath)
            if ext not in (".cpp", ".c", ".h", ".hpp", ".hxx", ".cxx"):
                continue
            if not self._should_lint_file(filepath):
                continue
            if os.path.exists(filepath):
                filtered_files.append(filepath)
        return filtered_files

    def _should_lint_file(self, filepath: str) -> bool:
        # TODO: Add more rules when enabling other projects to use clang-tidy in CI.
        return filepath.startswith("clang-tools-extra/clang-tidy/")

    def run_linter_tool(self, cpp_files: List[str], args: LintArgs) -> Optional[str]:
        if not cpp_files:
            return None

        git_diff_cmd = [
            "git",
            "diff",
            "-U0",
            f"{args.start_rev}...{args.end_rev}",
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
            return "Git diff failed"

        diff_content = diff_proc.stdout
        if not diff_content.strip():
            return None

        tidy_diff_cmd = [
            "clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py",
            "-path",
            args.build_path,
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

        clean_output = self._clean_clang_tidy_output(proc.stdout.strip())
        return clean_output

    def _clean_clang_tidy_output(self, output: str) -> Optional[str]:
        if not output or output == "No relevant changes found.":
            return None

        lines = output.split("\n")
        cleaned_lines = []

        for line in lines:
            if line.startswith("Running clang-tidy in") or line.endswith("generated."):
                continue

            idx = line.rfind("llvm-project/")
            if idx != -1:
                line = line[idx + len("llvm-project/") :]

            cleaned_lines.append(line)

        if cleaned_lines:
            return "\n".join(cleaned_lines)
        return None


class Doc8LintHelper(LintHelper):
    name = "doc8"
    friendly_name = "Documentation linter"

    def instructions(self, doc_files: List[str], args: LintArgs) -> str:
        files_str = " ".join(doc_files)
        return f"doc8 -q {files_str}"

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        filtered_files = []
        for filepath in changed_files:
            _, ext = os.path.splitext(filepath)
            if ext not in (".rst"):
                continue
            if not filepath.startswith("clang-tools-extra/docs/clang-tidy/checks/"):
                continue
            if os.path.exists(filepath):
                filtered_files.append(filepath)
        return filtered_files

    def run_linter_tool(self, doc_files: List[str], args: LintArgs) -> Optional[str]:
        if not doc_files:
            return None

        doc8_cmd = [args.doc8_binary, "-q"] + doc_files

        if args.verbose:
            print(f"Running doc8: {' '.join(doc8_cmd)}")

        proc = subprocess.run(
            doc8_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        output = proc.stdout.strip()
        if proc.returncode != 0 and not output:
            return proc.stderr.strip()


ALL_LINTERS = (ClangTidyLintHelper(), Doc8LintHelper())


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
        "--doc8-binary",
        type=str,
        default="doc8",
        help="Path to doc8 binary",
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True, help="Verbose output"
    )

    parsed_args = parser.parse_args()
    args = LintArgs(parsed_args)

    if args.verbose:
        print("Running all linters.")

    overall_success = True
    all_comments = []

    for linter in ALL_LINTERS:
        if args.verbose:
            print(f"Running linter: {linter.name}")

        linter_passed = linter.run(args)
        if not linter_passed:
            overall_success = False

        if linter.comment:
            all_comments.append(linter.comment)

    if len(all_comments):
        import json

        with open("comments", "w") as f:
            json.dump(all_comments, f)

    if not overall_success:
        print("error: Some linters failed.")
        sys.exit(1)
    else:
        print("All linters passed.")
