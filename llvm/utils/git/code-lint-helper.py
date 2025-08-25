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

    def __init__(self, args: argparse.Namespace = None) -> None:
        if not args is None:
            self.start_rev = args.start_rev
            self.end_rev = args.end_rev
            self.repo = args.repo
            self.token = args.token
            self.changed_files = args.changed_files
            self.issue_number = args.issue_number
            self.verbose = args.verbose
            self.build_path = args.build_path
            self.clang_tidy_binary = args.clang_tidy_binary


COMMENT_TAG = "<!--LLVM CODE LINT COMMENT: clang-tidy-->"


def get_instructions(cpp_files: List[str]) -> str:
    files_str = " ".join(cpp_files)
    return f"""
git diff -U0 origin/main...HEAD -- {files_str} |
python3 clang-tools-extra/clang-tidy/tool/clang-tidy-diff.py \\
  -path build -p1 -quiet"""


def clean_clang_tidy_output(output: str) -> Optional[str]:
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


# TODO: Add more rules when enabling other projects to use clang-tidy in CI.
def should_lint_file(filepath: str) -> bool:
    return filepath.startswith("clang-tools-extra/clang-tidy/")


def filter_changed_files(changed_files: List[str]) -> List[str]:
    filtered_files = []
    for filepath in changed_files:
        _, ext = os.path.splitext(filepath)
        if ext not in (".cpp", ".c", ".h", ".hpp", ".hxx", ".cxx"):
            continue
        if not should_lint_file(filepath):
            continue
        if os.path.exists(filepath):
            filtered_files.append(filepath)

    return filtered_files


def create_comment_text(warning: str, cpp_files: List[str]) -> str:
    instructions = get_instructions(cpp_files)
    return f"""
:warning: C/C++ code linter clang-tidy found issues in your code. :warning:

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
View the output from clang-tidy here.
</summary>

```
{warning}
```

</details>
"""


def find_comment(pr: any) -> any:
    for comment in pr.as_issue().get_comments():
        if COMMENT_TAG in comment.body:
            return comment
    return None


def create_comment(
    comment_text: str, args: LintArgs, create_new: bool
) -> Optional[dict]:
    import github

    repo = github.Github(args.token).get_repo(args.repo)
    pr = repo.get_issue(args.issue_number).as_pull_request()

    comment_text = COMMENT_TAG + "\n\n" + comment_text

    existing_comment = find_comment(pr)

    comment = None
    if create_new or existing_comment:
        comment = {"body": comment_text}
    if existing_comment:
        comment["id"] = existing_comment.id
    return comment


def run_clang_tidy(changed_files: List[str], args: LintArgs) -> Optional[str]:
    if not changed_files:
        print("no c/c++ files found")
        return None

    git_diff_cmd = [
        "git",
        "diff",
        "-U0",
        f"{args.start_rev}...{args.end_rev}",
        "--",
    ] + changed_files

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

    return clean_clang_tidy_output(proc.stdout.strip())


def run_linter(changed_files: List[str], args: LintArgs) -> tuple[bool, Optional[dict]]:
    changed_files = [arg for arg in changed_files if "third-party" not in arg]

    cpp_files = filter_changed_files(changed_files)

    tidy_result = run_clang_tidy(cpp_files, args)
    should_update_gh = args.token is not None and args.repo is not None

    comment = None
    if tidy_result is None:
        if should_update_gh:
            comment_text = (
                ":white_check_mark: With the latest revision "
                "this PR passed the C/C++ code linter."
            )
            comment = create_comment(comment_text, args, create_new=False)
        return True, comment
    elif len(tidy_result) > 0:
        if should_update_gh:
            comment_text = create_comment_text(tidy_result, cpp_files)
            comment = create_comment(comment_text, args, create_new=True)
        else:
            print(
                "Warning: C/C++ code linter, clang-tidy detected "
                "some issues with your code..."
            )
        return False, comment
    else:
        # The linter failed but didn't output a result (e.g. some sort of
        # infrastructure failure).
        comment_text = (
            ":warning: The C/C++ code linter failed without printing "
            "an output. Check the logs for output. :warning:"
        )
        comment = create_comment(comment_text, args, create_new=False)
        return False, comment


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

    if args.verbose:
        print("running linter clang-tidy")

    success, comment = run_linter(changed_files, args)

    if not success:
        if args.verbose:
            print("linter clang-tidy failed")

    # Write comments file if we have a comment
    if comment:
        if args.verbose:
            print(f"linter clang-tidy has comment: {comment}")

        with open("comments", "w") as f:
            import json

            json.dump([comment], f)

    if not success:
        print("error: some linters failed: clang-tidy")
        sys.exit(1)
