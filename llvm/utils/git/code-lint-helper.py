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
import github
import json
import os
import subprocess
import sys
from typing import Any, Dict, Final, List, Sequence


class LintArgs:
    start_rev: str
    end_rev: str
    repo: str
    changed_files: Sequence[str]
    token: str
    verbose: bool = True
    issue_number: int = 0
    build_path: str = "build"
    clang_tidy_binary: str = "clang-tidy"

    def __init__(self, args: argparse.Namespace) -> None:
        if args is not None:
            self.start_rev = args.start_rev
            self.end_rev = args.end_rev
            self.repo = args.repo
            self.token = args.token
            self.changed_files = (
                args.changed_files.split(",") if args.changed_files else []
            )
            self.issue_number = args.issue_number
            self.verbose = args.verbose
            self.build_path = args.build_path
            self.clang_tidy_binary = args.clang_tidy_binary


class LintHelper:
    COMMENT_TAG: Final = "<!--LLVM CODE LINT COMMENT: {linter}-->"
    name: str
    friendly_name: str
    comment: Dict[str, Any] = {}

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG.format(linter=self.name)

    def instructions(self, files_to_lint: Sequence[str], args: LintArgs) -> str:
        raise NotImplementedError()

    def filter_changed_files(self, changed_files: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError()

    def run_linter_tool(self, files_to_lint: Sequence[str], args: LintArgs) -> str:
        raise NotImplementedError()

    def create_comment_text(
        self, linter_output: str, files_to_lint: Sequence[str], args: LintArgs
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

    def find_comment(self, pr: Any) -> Any:
        for comment in pr.as_issue().get_comments():
            if comment.body.startswith(self.comment_tag):
                return comment
        return None

    def update_pr(self, comment_text: str, args: LintArgs, create_new: bool) -> None:
        assert args.repo is not None
        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        comment_text = f"{self.comment_tag}\n\n{comment_text}"

        existing_comment = self.find_comment(pr)

        if existing_comment:
            self.comment = {"body": comment_text, "id": existing_comment.id}
        elif create_new:
            self.comment = {"body": comment_text}


    def run(self, args: LintArgs) -> bool:
        if args.verbose:
            print(f"got changed files: {args.changed_files}")

        files_to_lint = self.filter_changed_files(args.changed_files)

        if not files_to_lint and args.verbose:
            print("no modified files found")

        is_success = True
        linter_output = ""

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
                    comment_text = self.create_comment_text(
                        linter_output, files_to_lint, args
                    )
                    self.update_pr(comment_text, args, create_new=True)
                else:
                    # The linter failed but didn't output a result (e.g. some sort of
                    # infrastructure failure).
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
    name: Final = "clang-tidy"
    friendly_name: Final = "C/C++ code linter"

    def instructions(self, files_to_lint: Sequence[str], args: LintArgs) -> str:
        files_str = " ".join(files_to_lint)
        return f"""
git diff -U0 origin/main...HEAD -- {files_str} |
python3 clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py \
  -path {args.build_path} -p1 -quiet"""

    def filter_changed_files(self, changed_files: Sequence[str]) -> Sequence[str]:
        clang_tidy_changed_files = [
            arg for arg in changed_files if "third-party" not in arg
        ]

        filtered_files = []
        for filepath in clang_tidy_changed_files:
            _, ext = os.path.splitext(filepath)
            if ext not in (".c", ".cpp", ".cxx", ".h", ".hpp", ".hxx"):
                continue
            if not self._should_lint_file(filepath):
                continue
            if os.path.exists(filepath):
                filtered_files.append(filepath)
        return filtered_files

    def _should_lint_file(self, filepath: str) -> bool:
        # TODO: Add more rules when enabling other projects to use clang-tidy in CI.
        return filepath.startswith("clang-tools-extra/clang-tidy/")

    def run_linter_tool(self, files_to_lint: Sequence[str], args: LintArgs) -> str:
        if not files_to_lint:
            return ""

        git_diff_cmd = [
            "git",
            "diff",
            "-U0",
            f"{args.start_rev}...{args.end_rev}",
            "--",
        ]
        git_diff_cmd.extend(files_to_lint)

        diff_proc = subprocess.run(
            git_diff_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if diff_proc.returncode != 0:
            print(f"Git diff failed: {diff_proc.stderr}")
            return ""

        diff_content = diff_proc.stdout
        if not diff_content.strip():
            if args.verbose:
                print("No diff content found")
            return ""

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

    def _clean_clang_tidy_output(self, output: str) -> str:
        """
        - Remove 'Running clang-tidy in X threads...' line
        - Remove 'N warnings generated.' line
        - Strip leading workspace path from file paths
        """
        if not output or output == "No relevant changes found.":
            return ""

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
        return ""



ALL_LINTERS = (ClangTidyLintHelper(),)


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

    failed_linters: List[str] = []
    comments: List[Dict[str, Any]] = []

    for linter in ALL_LINTERS:
        if args.verbose:
            print(f"running linter {linter.name}")

        if not linter.run(args):
            failed_linters.append(linter.name)
            if args.verbose:
                print(f"linter {linter.name} failed")

        # Write comments file if we have a comment
        if linter.comment:
            comments.append(linter.comment)
            if args.verbose:
                print(f"linter {linter.name} has comment: {linter.comment}")

    if len(comments):
        with open("comments", "w") as f:
            json.dump(comments, f)

    if len(failed_linters) > 0:
        print(f"error: some linters failed: {' '.join(failed_linters)}")
        sys.exit(1)
