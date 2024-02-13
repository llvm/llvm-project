# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tool for counting the number of currently running Github actions jobs.

This tool counts and enumerates the currently active jobs in Github actions
for the monorepo.

python3 ./count_running_jobs.py --token=<github token>

Note that the token argument is optional. If it is not specified, the queries
will be performed unauthenticated.
"""

import argparse
import github


def main(token):
    workflows = (
        github.Github(args.token)
        .get_repo("llvm/llvm-project")
        .get_workflow_runs(status="in_progress")
    )

    in_progress_jobs = 0

    for workflow in workflows:
        for job in workflow.jobs():
            if job.status == "in_progress":
                print(f"{workflow.name}/{job.name}")
                in_progress_jobs += 1

    print(f"\nFound {in_progress_jobs} running jobs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A tool for listing and counting Github actions jobs"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="The Github token to use to authorize with the API",
        default=None,
        nargs="?",
    )
    args = parser.parse_args()
    main(args.token)
