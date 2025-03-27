#!/usr/bin/env python3
# ===-- merge-release-pr.py  ------------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#

"""
Helper script that will merge a Pull Request into a release branch. It will first
do some validations of the PR then rebase and finally push the changes to the
release branch.

Usage: merge-release-pr.py <PR id>
By default it will push to the 'upstream' origin, but you can pass
--upstream-origin/-o <origin> if you want to change it.

If you want to skip a specific validation, like the status checks you can
pass -s status_checks, this argument can be passed multiple times.
"""

import argparse
import json
import subprocess
import sys
import time
from typing import List


class PRMerger:
    def __init__(self, args):
        self.args = args

    def run_gh(self, gh_cmd: str, args: List[str]) -> str:
        cmd = ["gh", gh_cmd, "-Rllvm/llvm-project"] + args
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            print(p.stderr)
            raise RuntimeError("Failed to run gh")
        return p.stdout

    def validate_state(self, data):
        """Validate the state of the PR, this means making sure that it is OPEN and not already merged or closed."""
        state = data["state"]
        if state != "OPEN":
            return False, f"state is {state.lower()}, not open"
        return True

    def validate_target_branch(self, data):
        """
        Validate that the PR is targetting a release/ branch. We could
        validate the exact branch here, but I am not sure how to figure
        out what we want except an argument and that might be a bit to
        to much overhead.
        """
        baseRefName: str = data["baseRefName"]
        if not baseRefName.startswith("release/"):
            return False, f"target branch is {baseRefName}, not a release branch"
        return True

    def validate_approval(self, data):
        """
        Validate the approval decision. This checks that the PR has been
        approved.
        """
        if data["reviewDecision"] != "APPROVED":
            return False, "PR is not approved"
        return True

    def validate_status_checks(self, data):
        """
        Check that all the actions / status checks succeeded. Will also
        fail if we have status checks in progress.
        """
        failures = []
        pending = []
        for status in data["statusCheckRollup"]:
            if "conclusion" in status and status["conclusion"] == "FAILURE":
                failures.append(status)
            if "status" in status and status["status"] == "IN_PROGRESS":
                pending.append(status)

        if failures or pending:
            errstr = "\n"
            if failures:
                errstr += "    FAILED: "
                errstr += ", ".join([d["name"] for d in failures])
            if pending:
                if failures:
                    errstr += "\n"
                errstr += "    PENDING: "
                errstr += ", ".join([d["name"] for d in pending])

            return False, errstr

        return True

    def validate_commits(self, data):
        """
        Validate that the PR contains just one commit. If it has more
        we might want to squash. Which is something we could add to
        this script in the future.
        """
        if len(data["commits"]) > 1:
            return False, f"More than 1 commit! {len(data['commits'])}"
        return True

    def _normalize_pr(self, parg: str):
        if parg.isdigit():
            return parg
        elif parg.startswith("https://github.com/llvm/llvm-project/pull"):
            # try to parse the following url https://github.com/llvm/llvm-project/pull/114089
            i = parg[parg.rfind("/") + 1 :]
            if not i.isdigit():
                raise RuntimeError(f"{i} is not a number, malformatted input.")
            return i
        else:
            raise RuntimeError(
                f"PR argument must be PR ID or pull request URL - {parg} is wrong."
            )

    def load_pr_data(self):
        self.args.pr = self._normalize_pr(self.args.pr)
        fields_to_fetch = [
            "baseRefName",
            "commits",
            "headRefName",
            "headRepository",
            "headRepositoryOwner",
            "reviewDecision",
            "state",
            "statusCheckRollup",
            "title",
            "url",
        ]
        print(f"> Loading PR {self.args.pr}...")
        o = self.run_gh(
            "pr",
            ["view", self.args.pr, "--json", ",".join(fields_to_fetch)],
        )
        self.prdata = json.loads(o)

        # save the baseRefName (target branch) so that we know where to push
        self.target_branch = self.prdata["baseRefName"]
        srepo = self.prdata["headRepository"]["name"]
        sowner = self.prdata["headRepositoryOwner"]["login"]
        self.source_url = f"https://github.com/{sowner}/{srepo}"
        self.source_branch = self.prdata["headRefName"]

        if srepo != "llvm-project":
            print("The target repo is NOT llvm-project, check the PR!")
            sys.exit(1)

        if sowner == "llvm":
            print(
                "The source owner should never be github.com/llvm, double check the PR!"
            )
            sys.exit(1)

    def validate_pr(self):
        print(f"> Handling PR {self.args.pr} - {self.prdata['title']}")
        print(f">   {self.prdata['url']}")

        VALIDATIONS = {
            "state": self.validate_state,
            "target_branch": self.validate_target_branch,
            "approval": self.validate_approval,
            "commits": self.validate_commits,
            "status_checks": self.validate_status_checks,
        }

        print()
        print("> Validations:")
        total_ok = True
        for val_name, val_func in VALIDATIONS.items():
            try:
                validation_data = val_func(self.prdata)
            except:
                validation_data = False
            ok = None
            skipped = (
                True
                if (self.args.skip_validation and val_name in self.args.skip_validation)
                else False
            )
            if isinstance(validation_data, bool) and validation_data:
                ok = "OK"
            elif isinstance(validation_data, tuple) and not validation_data[0]:
                failstr = validation_data[1]
                if skipped:
                    ok = "SKIPPED: "
                else:
                    total_ok = False
                    ok = "FAIL: "
                ok += failstr
            else:
                ok = "FAIL! (Unknown)"
            print(f"  * {val_name}: {ok}")
        return total_ok

    def rebase_pr(self):
        print("> Fetching upstream")
        subprocess.run(["git", "fetch", "--all"], check=True)
        print("> Rebasing...")
        subprocess.run(
            ["git", "rebase", self.args.upstream + "/" + self.target_branch], check=True
        )
        print("> Publish rebase...")
        subprocess.run(
            ["git", "push", "--force", self.source_url, f"HEAD:{self.source_branch}"]
        )

    def checkout_pr(self):
        print("> Fetching PR changes...")
        self.merge_branch = "llvm_merger_" + self.args.pr
        self.run_gh(
            "pr",
            [
                "checkout",
                self.args.pr,
                "--force",
                "--branch",
                self.merge_branch,
            ],
        )

        # get the branch information so that we can use it for
        # pushing later.
        p = subprocess.run(
            ["git", "config", f"branch.{self.merge_branch}.merge"],
            check=True,
            capture_output=True,
            text=True,
        )
        upstream_branch = p.stdout.strip().replace("refs/heads/", "")
        print(upstream_branch)

    def push_upstream(self):
        print("> Pushing changes...")
        subprocess.run(
            ["git", "push", self.args.upstream, "HEAD:" + self.target_branch],
            check=True,
        )

    def delete_local_branch(self):
        print("> Deleting the old branch...")
        subprocess.run(["git", "switch", "main"])
        subprocess.run(["git", "branch", "-D", f"llvm_merger_{self.args.pr}"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pr",
        help="The Pull Request ID that should be merged into a release. Can be number or URL",
    )
    parser.add_argument(
        "--skip-validation",
        "-s",
        action="append",
        help="Skip a specific validation, can be passed multiple times. I.e. -s status_checks -s approval",
    )
    parser.add_argument(
        "--upstream-origin",
        "-o",
        default="upstream",
        dest="upstream",
        help="The name of the origin that we should push to. (default: upstream)",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Run validations, rebase and fetch, but don't push.",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only run the validations."
    )
    parser.add_argument(
        "--rebase-only", action="store_true", help="Only rebase and exit"
    )
    args = parser.parse_args()

    merger = PRMerger(args)
    merger.load_pr_data()

    if args.rebase_only:
        merger.checkout_pr()
        merger.rebase_pr()
        merger.delete_local_branch()
        sys.exit(0)

    if not merger.validate_pr():
        print()
        print(
            "! Validations failed! Pass --skip-validation/-s <validation name> to pass this, can be passed multiple times"
        )
        sys.exit(1)

    if args.validate_only:
        print()
        print("! --validate-only passed, will exit here")
        sys.exit(0)

    merger.checkout_pr()
    merger.rebase_pr()

    if args.no_push:
        print()
        print("! --no-push passed, will exit here")
        sys.exit(0)

    merger.push_upstream()
    merger.delete_local_branch()

    print()
    print("> Done! Have a nice day!")
