#!/usr/bin/env python3
#
# ====- email-check-helper, checks for private email usage in PRs --*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==--------------------------------------------------------------------------------------==#
"""A helper script to detect private email of a Github user
This script is run by GitHub actions to ensure that contributors to PR's are not
using GitHub's private email addresses.

The script enforces the LLVM Developer Policy regarding email addresses:
https://llvm.org/docs/DeveloperPolicy.html#email-addresses
"""

import argparse
import json
import os
import subprocess
import sys
from typing import Optional


COMMENT_TAG = "<!--LLVM EMAIL CHECK COMMENT-->"


def get_commit_email() -> Optional[str]:
    proc = subprocess.run(
        ["git", "show", "-s", "--format=%ae", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        check=False
    )
    if proc.returncode == 0:
        return proc.stdout.strip()
    return None


def is_private_email(email: Optional[str]) -> bool:
    if not email:
        return False
    return (email.endswith("noreply.github.com") or
            email.endswith("users.noreply.github.com"))


def check_user_email(token: str, pr_author: str) -> bool:
    try:
        from github import Github

        print(f"Checking email privacy for user: {pr_author}")
        
        api = Github(token)
        user = api.get_user(pr_author)
        emails = user.get_emails()
        print(emails)
        
        print(f"User public email: {user.email or 'null (private)'}")

        if user.email is not None or is_private_email(user.email):
            return True

        return is_private_email(get_commit_email())
    except Exception as e:
        print(f"got exception {e.with_traceback()}")
        return False


def generate_comment() -> str:
    return f"""{COMMENT_TAG}
⚠️ We detected that you are using a GitHub private e-mail address to contribute to the repo.<br/>
Please turn off [Keep my email addresses private](https://github.com/settings/emails) setting in your account.<br/>
See [LLVM Developer Policy](https://llvm.org/docs/DeveloperPolicy.html#email-addresses) and
[LLVM Discourse](https://discourse.llvm.org/t/hidden-emails-on-github-should-we-do-something-about-it) for more information.
"""


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Check for private email usage in GitHub PRs"
    )
    parser.add_argument(
        "--token", type=str, required=True, help="GitHub authentication token"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=os.getenv("GITHUB_REPOSITORY", "llvm/llvm-project"),
        help="The GitHub repository in the form of <owner>/<repo>",
    )
    parser.add_argument(
        "--pr-author",
        type=str,
        required=True,
        help="The GitHub username of the PR author"
    )

    args = parser.parse_args()

    has_private_email = check_user_email(args.token, args.pr_author)

    comments = []
    if has_private_email:
        comments.append({"body": generate_comment()})

    with open("comments", "w", encoding="utf-8") as f:
        json.dump(comments, f)

    print(f"Wrote {'comment' if has_private_email else 'empty comments'} to file")

    if has_private_email:
        print("Private email detected")
        sys.exit(1)


if __name__ == "__main__":
    main()
