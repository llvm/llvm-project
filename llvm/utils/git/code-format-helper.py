#!/usr/bin/env python3
#
# ====- code-format-helper, runs code formatters from the ci --*- python -*--==#
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
from functools import cached_property

import github
from github import IssueComment, PullRequest


class FormatHelper:
    COMMENT_TAG = "<!--LLVM CODE FORMAT COMMENT: {fmt}-->"
    name = "unknown"

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG.replace("fmt", self.name)

    def format_run(self, changed_files: [str], args: argparse.Namespace) -> str | None:
        pass

    def pr_comment_text(self, diff: str) -> str:
        return f"""
{self.comment_tag}

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

    def find_comment(
        self, pr: PullRequest.PullRequest
    ) -> IssueComment.IssueComment | None:
        for comment in pr.as_issue().get_comments():
            if self.comment_tag in comment.body:
                return comment
        return None

    def update_pr(self, diff: str, args: argparse.Namespace):
        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        existing_comment = self.find_comment(pr)
        pr_text = self.pr_comment_text(diff)

        if existing_comment:
            existing_comment.edit(pr_text)
        else:
            pr.as_issue().create_comment(pr_text)

    def update_pr_success(self, args: argparse.Namespace):
        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        existing_comment = self.find_comment(pr)
        if existing_comment:
            existing_comment.edit(
                f"""
{self.comment_tag}
:white_check_mark: With the latest revision this PR passed the {self.friendly_name}.
"""
            )

    def run(self, changed_files: [str], args: argparse.Namespace):
        diff = self.format_run(changed_files, args)
        if diff:
            self.update_pr(diff, args)
            return False
        else:
            self.update_pr_success(args)
            return True


class ClangFormatHelper(FormatHelper):
    name = "clang-format"
    friendly_name = "C/C++ code formatter"

    @property
    def instructions(self):
        return " ".join(self.cf_cmd)

    @cached_property
    def libcxx_excluded_files(self):
        with open("libcxx/utils/data/ignore_format.txt", "r") as ifd:
            return [excl.strip() for excl in ifd.readlines()]

    def should_be_excluded(self, path: str) -> bool:
        if path in self.libcxx_excluded_files:
            print(f"Excluding file {path}")
            return True
        return False

    def filter_changed_files(self, changed_files: [str]) -> [str]:
        filtered_files = []
        for path in changed_files:
            _, ext = os.path.splitext(path)
            if ext in (".cpp", ".c", ".h", ".hpp", ".hxx", ".cxx"):
                if not self.should_be_excluded(path):
                    filtered_files.append(path)
        return filtered_files

    def format_run(self, changed_files: [str], args: argparse.Namespace) -> str | None:
        cpp_files = self.filter_changed_files(changed_files)
        if not cpp_files:
            return
        cf_cmd = [
            "git-clang-format",
            "--diff",
            args.start_rev,
            args.end_rev,
            "--",
        ] + cpp_files
        print(f"Running: {' '.join(cf_cmd)}")
        self.cf_cmd = cf_cmd
        proc = subprocess.run(cf_cmd, capture_output=True)

        # formatting needed
        if proc.returncode == 1:
            return proc.stdout.decode("utf-8")

        return None


class DarkerFormatHelper(FormatHelper):
    name = "darker"
    friendly_name = "Python code formatter"

    @property
    def instructions(self):
        return " ".join(self.darker_cmd)

    def filter_changed_files(self, changed_files: [str]) -> [str]:
        filtered_files = []
        for path in changed_files:
            name, ext = os.path.splitext(path)
            if ext == ".py":
                filtered_files.append(path)

        return filtered_files

    def format_run(self, changed_files: [str], args: argparse.Namespace) -> str | None:
        py_files = self.filter_changed_files(changed_files)
        if not py_files:
            return
        darker_cmd = [
            "darker",
            "--check",
            "--diff",
            "-r",
            f"{args.start_rev}..{args.end_rev}",
        ] + py_files
        print(f"Running: {' '.join(darker_cmd)}")
        self.darker_cmd = darker_cmd
        proc = subprocess.run(darker_cmd, capture_output=True)

        # formatting needed
        if proc.returncode == 1:
            return proc.stdout.decode("utf-8")

        return None


ALL_FORMATTERS = (DarkerFormatHelper(), ClangFormatHelper())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token", type=str, required=True, help="GitHub authentiation token"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", "llvm/llvm-project"),
        help="The GitHub repository that we are working with in the form of <owner>/<repo> (e.g. llvm/llvm-project)",
    )
    parser.add_argument("--issue-number", type=int, required=True)
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

    args = parser.parse_args()

    changed_files = []
    if args.changed_files:
        changed_files = args.changed_files.split(",")

    exit_code = 0
    for fmt in ALL_FORMATTERS:
        if not fmt.run(changed_files, args):
            exit_code = 1

    sys.exit(exit_code)
