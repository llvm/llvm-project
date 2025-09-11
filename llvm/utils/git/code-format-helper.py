#!/usr/bin/env python3
#
# ====- code-format-helper, runs code formatters from the ci or in a hook --*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==--------------------------------------------------------------------------------------==#

import argparse
import os
import re
import shlex
import subprocess
import sys
from typing import List, Optional

"""
This script is run by GitHub actions to ensure that the code in PR's conform to
the coding style of LLVM. It can also be installed as a pre-commit git hook to
check the coding style before submitting it. The canonical source of this script
is in the LLVM source tree under llvm/utils/git.

For C/C++ code it uses clang-format and for Python code it uses darker (which
in turn invokes black).

You can learn more about the LLVM coding style on llvm.org:
https://llvm.org/docs/CodingStandards.html

You can install this script as a git hook by symlinking it to the .git/hooks
directory:

ln -s $(pwd)/llvm/utils/git/code-format-helper.py .git/hooks/pre-commit

You can control the exact path to clang-format or darker with the following
environment variables: $CLANG_FORMAT_PATH and $DARKER_FORMAT_PATH.
"""


class FormatArgs:
    start_rev: str = None
    end_rev: str = None
    repo: str = None
    changed_files: List[str] = []
    token: str = None
    verbose: bool = True
    issue_number: int = 0
    write_comment_to_file: bool = False

    def __init__(self, args: argparse.Namespace = None) -> None:
        if not args is None:
            self.start_rev = args.start_rev
            self.end_rev = args.end_rev
            self.repo = args.repo
            self.token = args.token
            self.changed_files = args.changed_files
            self.issue_number = args.issue_number
            self.write_comment_to_file = args.write_comment_to_file


class FormatHelper:
    COMMENT_TAG = "<!--LLVM CODE FORMAT COMMENT: {fmt}-->"
    name: str
    friendly_name: str
    comment: dict = None

    @property
    def comment_tag(self) -> str:
        return self.COMMENT_TAG.replace("fmt", self.name)

    @property
    def instructions(self) -> str:
        raise NotImplementedError()

    def has_tool(self) -> bool:
        raise NotImplementedError()

    def format_run(self, changed_files: List[str], args: FormatArgs) -> Optional[str]:
        raise NotImplementedError()

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

    # TODO: any type should be replaced with the correct github type, but it requires refactoring to
    # not require the github module to be installed everywhere.
    def find_comment(self, pr: any) -> any:
        for comment in pr.as_issue().get_comments():
            if self.comment_tag in comment.body:
                return comment
        return None

    def update_pr(self, comment_text: str, args: FormatArgs, create_new: bool) -> None:
        import github
        from github import IssueComment, PullRequest

        repo = github.Github(args.token).get_repo(args.repo)
        pr = repo.get_issue(args.issue_number).as_pull_request()

        comment_text = self.comment_tag + "\n\n" + comment_text

        existing_comment = self.find_comment(pr)

        if args.write_comment_to_file:
            if create_new or existing_comment:
                self.comment = {"body": comment_text}
            if existing_comment:
                self.comment["id"] = existing_comment.id
            return

        if existing_comment:
            existing_comment.edit(comment_text)
        elif create_new:
            pr.as_issue().create_comment(comment_text)

    def run(self, changed_files: List[str], args: FormatArgs) -> bool:
        changed_files = [arg for arg in changed_files if "third-party" not in arg]
        diff = self.format_run(changed_files, args)
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
                comment_text = self.pr_comment_text_for_diff(diff)
                self.update_pr(comment_text, args, create_new=True)
            else:
                print(
                    f"Warning: {self.friendly_name}, {self.name} detected "
                    "some issues with your code formatting..."
                )
            return False
        else:
            # The formatter failed but didn't output a diff (e.g. some sort of
            # infrastructure failure).
            comment_text = (
                f":warning: The {self.friendly_name} failed without printing "
                "a diff. Check the logs for stderr output. :warning:"
            )
            self.update_pr(comment_text, args, create_new=False)
            return False


class ClangFormatHelper(FormatHelper):
    name = "clang-format"
    friendly_name = "C/C++ code formatter"

    def _construct_command(self, diff_expression: list[str] | None):
        cf_cmd = [self.clang_fmt_path, "--diff"]

        if diff_expression:
            cf_cmd.extend(diff_expression)

        # Gather the extension of all modified files and pass them explicitly to git-clang-format.
        # This prevents git-clang-format from applying its own filtering rules on top of ours.
        extensions = set()
        for file in self._cpp_files:
            _, ext = os.path.splitext(file)
            extensions.add(
                ext.strip(".")
            )  # Exclude periods since git-clang-format takes extensions without them
        cf_cmd.append("--extensions")
        cf_cmd.append(",".join(extensions))

        cf_cmd.append("--")
        cf_cmd += self._cpp_files
        return cf_cmd

    @property
    def instructions(self) -> str:
        # TODO(boomanaiden154): Add --diff_from_common_commit option when it has
        # landed as in available in a released version.
        return " ".join(self._construct_command(["origin/main", "HEAD"]))

    def should_include_extensionless_file(self, path: str) -> bool:
        return path.startswith("libcxx/include")

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        filtered_files = []
        for path in changed_files:
            _, ext = os.path.splitext(path)
            if ext in (
                ".cpp",
                ".c",
                ".h",
                ".hpp",
                ".hxx",
                ".cxx",
                ".inc",
                ".cppm",
                ".cl",
            ):
                filtered_files.append(path)
            elif ext == "" and self.should_include_extensionless_file(path):
                filtered_files.append(path)
        return filtered_files

    @property
    def clang_fmt_path(self) -> str:
        if "CLANG_FORMAT_PATH" in os.environ:
            return os.environ["CLANG_FORMAT_PATH"]
        return "git-clang-format"

    def has_tool(self) -> bool:
        cmd = [self.clang_fmt_path, "-h"]
        proc = None
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            return False
        return proc.returncode == 0

    def format_run(self, changed_files: List[str], args: FormatArgs) -> Optional[str]:
        self._cpp_files = self.filter_changed_files(changed_files)
        if not self._cpp_files:
            return None

        diff_expression = []
        if args.start_rev and args.end_rev:
            diff_expression.append(args.start_rev)
            diff_expression.append(args.end_rev)

        cf_cmd = self._construct_command(diff_expression)

        if args.verbose:
            print(f"Running: {' '.join(cf_cmd)}")
        proc = subprocess.run(cf_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.stdout.write(proc.stderr.decode("utf-8"))

        if proc.returncode != 0:
            # formatting needed, or the command otherwise failed
            if args.verbose:
                print(f"error: {self.name} exited with code {proc.returncode}")
                # Print the diff in the log so that it is viewable there
                print(proc.stdout.decode("utf-8"))
            return proc.stdout.decode("utf-8")
        else:
            return None


class DarkerFormatHelper(FormatHelper):
    name = "darker"
    friendly_name = "Python code formatter"

    def _construct_command(self, diff_expression: str | None) -> str:
        darker_cmd = [
            self.darker_fmt_path,
            "--check",
            "--diff",
        ]
        if diff_expression:
            darker_cmd += ["-r", diff_expression]
        darker_cmd += self._py_files
        return darker_cmd

    @property
    def instructions(self) -> str:
        return " ".join(self._construct_command("origin/main...HEAD"))

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        filtered_files = []
        for path in changed_files:
            name, ext = os.path.splitext(path)
            if ext == ".py":
                filtered_files.append(path)

        return filtered_files

    @property
    def darker_fmt_path(self) -> str:
        if "DARKER_FORMAT_PATH" in os.environ:
            return os.environ["DARKER_FORMAT_PATH"]
        return "darker"

    def has_tool(self) -> bool:
        cmd = [self.darker_fmt_path, "--version"]
        proc = None
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except:
            return False
        return proc.returncode == 0

    def format_run(self, changed_files: List[str], args: FormatArgs) -> Optional[str]:
        py_files = self.filter_changed_files(changed_files)
        if not py_files:
            return None
        self._py_files = py_files
        diff_expression = None
        if args.start_rev and args.end_rev:
            diff_expression = f"{args.start_rev}...{args.end_rev}"
        darker_cmd = self._construct_command(diff_expression)
        if args.verbose:
            print(f"Running: {' '.join(darker_cmd)}")
        proc = subprocess.run(
            darker_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if args.verbose:
            sys.stdout.write(proc.stderr.decode("utf-8"))

        if proc.returncode != 0:
            # formatting needed, or the command otherwise failed
            if args.verbose:
                print(f"error: {self.name} exited with code {proc.returncode}")
                # Print the diff in the log so that it is viewable there
                print(proc.stdout.decode("utf-8"))
            return proc.stdout.decode("utf-8")
        else:
            sys.stdout.write(proc.stdout.decode("utf-8"))
            return None


class UndefGetFormatHelper(FormatHelper):
    name = "undef deprecator"
    friendly_name = "undef deprecator"

    @property
    def instructions(self) -> str:
        return " ".join(shlex.quote(c) for c in self.cmd)

    def filter_changed_files(self, changed_files: List[str]) -> List[str]:
        filtered_files = []
        for path in changed_files:
            _, ext = os.path.splitext(path)
            if ext in (".cpp", ".c", ".h", ".hpp", ".hxx", ".cxx", ".inc", ".cppm", ".ll"):
                filtered_files.append(path)
        return filtered_files

    def has_tool(self) -> bool:
        return True

    def pr_comment_text_for_diff(self, diff: str) -> str:
        return f"""
:warning: {self.name} found issues in your code. :warning:

<details>
<summary>
You can test this locally with the following command:
</summary>

``````````bash
{self.instructions}
``````````

</details>

{diff}
"""

    def format_run(self, changed_files: List[str], args: FormatArgs) -> Optional[str]:
        files = self.filter_changed_files(changed_files)
        if not files:
            return None

        # Use git to find files that have had a change in the number of undefs
        regex = "([^a-zA-Z0-9#_-]undef[^a-zA-Z0-9_-]|UndefValue::get)"
        cmd = ["git", "diff", "-U0", "--pickaxe-regex", "-S", regex]

        if args.start_rev and args.end_rev:
            cmd.append(args.start_rev)
            cmd.append(args.end_rev)

        cmd += files
        self.cmd = cmd

        if args.verbose:
            print(f"Running: {self.instructions}")

        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8"
        )
        sys.stdout.write(proc.stderr)
        stdout = proc.stdout

        if not stdout:
            return None

        files = []

        # Split the diff so we have one array entry per file.
        # Each file is prefixed like:
        # diff --git a/file b/file
        for file in re.split("^diff --git ", stdout, 0, re.MULTILINE):
            lines = file.splitlines()
            match = re.match("a/([^ ]+)", lines[0] if lines else "")
            filename = match[1] if match else ""
            if filename.endswith(".ll"):
                undef_regex = r"(?<!%)\bundef\b"
            else:
                undef_regex = r"UndefValue::get"
            # search for additions of undef
            if re.search(r"^[+].*" + undef_regex, file, re.MULTILINE):
                files.append(filename)

        if not files:
            return None

        files = "\n".join(" - " + f for f in files)
        report = f"""
The following files introduce new uses of undef:
{files}

[Undef](https://llvm.org/docs/LangRef.html#undefined-values) is now deprecated and should only be used in the rare cases where no replacement is possible. For example, a load of uninitialized memory yields `undef`. You should use `poison` values for placeholders instead.

In tests, avoid using `undef` and having tests that trigger undefined behavior. If you need an operand with some unimportant value, you can add a new argument to the function and use that instead.

For example, this is considered a bad practice:
```llvm
define void @fn() {{
  ...
  br i1 undef, ...
}}
```

Please use the following instead:
```llvm
define void @fn(i1 %cond) {{
  ...
  br i1 %cond, ...
}}
```

Please refer to the [Undefined Behavior Manual](https://llvm.org/docs/UndefinedBehavior.html) for more information.
"""
        if args.verbose:
            print(f"error: {self.name} failed")
            print(report)
        return report


ALL_FORMATTERS = (DarkerFormatHelper(), ClangFormatHelper(), UndefGetFormatHelper())


def hook_main():
    # fill out args
    args = FormatArgs()
    args.verbose = os.getenv("FORMAT_HOOK_VERBOSE", False)

    # find the changed files
    cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=d"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = proc.stdout.decode("utf-8")
    for line in output.splitlines():
        args.changed_files.append(line)

    failed_fmts = []
    for fmt in ALL_FORMATTERS:
        if fmt.has_tool():
            if not fmt.run(args.changed_files, args):
                failed_fmts.append(fmt.name)
            if fmt.comment:
                comments.append(fmt.comment)
        else:
            print(f"Couldn't find {fmt.name}, can't check " + fmt.friendly_name.lower())

    if len(failed_fmts) > 0:
        print(
            "Pre-commit format hook failed, rerun with FORMAT_HOOK_VERBOSE=1 environment for verbose output"
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    script_path = os.path.abspath(__file__)
    if ".git/hooks" in script_path:
        hook_main()
        sys.exit(0)

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
    parser.add_argument(
        "--write-comment-to-file",
        action="store_true",
        help="Don't post comments on the PR, instead write the comments and metadata a file called 'comment'",
    )

    args = FormatArgs(parser.parse_args())

    changed_files = []
    if args.changed_files:
        changed_files = args.changed_files.split(",")

    failed_formatters = []
    comments = []
    for fmt in ALL_FORMATTERS:
        if not fmt.run(changed_files, args):
            failed_formatters.append(fmt.name)
        if fmt.comment:
            comments.append(fmt.comment)

    if len(comments):
        with open("comments", "w") as f:
            import json

            json.dump(comments, f)

    if len(failed_formatters) > 0:
        print(f"error: some formatters failed: {' '.join(failed_formatters)}")
        sys.exit(1)
