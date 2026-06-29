#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##
#
# This script is meant to be used by LLVM security response team members to
# help make sure transparency reports cover all security issues that were
# raised.
# It fetches the list of security advisories from a GitHub repository and
# compares it with a list of covered advisories in the transparency report.
# It prints out the uncovered advisories that are either closed or published.
# To run this script, you need to have the `gh` CLI installed and
# authenticated with your GitHub account.
#
# Typically, this script will be invoked as follows:
# $ python find_missing_sg_transparency_report_entries.py \
#     --owner llvm \
#     --repo llvm-security-repo \
#     --transparency-report llvm-project/llvm/docs/SecurityTransparencyReports.rst
#
# The script assumes that the transparency report has a format with enumerated items
# to cover issues that were raised. It assumes that the enumerated items have the
# following structure:
#
# _\d_. _Summary for the issue, on a single line_
#       _One or more GHSA IDs, possibly on multiple lines, but no other text in between_
#       _More text describing the issue_
#
#
# An example script output is as follows:
#
# 1. GHSA-q6fr-rhw7-35gh (closed): [Not a new security issue] Continued discussion for GHSA-w7qc-292v-5xh6
#   URL: https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-q6fr-rhw7-35gh
#   CVE: None
#   Created: 2024-07-19T07:56:09Z
#   Updated: 2024-09-23T10:01:00Z
#   Closed: 2024-09-23T10:01:00Z
#
# 2. GHSA-wh65-j229-6wfp (published): CMSE secure state may leak from stack to floating-point registers
#   URL: https://github.com/llvm/llvm-security-repo/security/advisories/GHSA-wh65-j229-6wfp
#   CVE: CVE-2024-7883
#   Created: 2024-08-27T13:21:31Z
#   Updated: 2025-02-04T13:42:08Z
#   Closed: None
#
#
# The unittests can be run by passing the `--test` argument to the script:
#   python find_missing_sg_transparency_report_entries.py --test

import argparse
from datetime import datetime
import re
import sys
import subprocess
import json


def fetch_advisory_ids(owner, repo):
    advisories = []
    page = 1

    while True:
        cmd = [
            "gh",
            "api",
            f"/repos/{owner}/{repo}/security-advisories",
            "--paginate",
            "--jq",
            ".[] | {ghsa_id: .ghsa_id, state: .state, summary: .summary, "
            + "html_url: .html_url, cve_id: .cve_id, created_at: .created_at, "
            + "updated_at: .updated_at, closed_at: .closed_at}",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            print(
                "Error: 'gh' CLI tool is not installed or not in PATH.", file=sys.stderr
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print("Error running gh CLI:", e.stderr, file=sys.stderr)
            sys.exit(1)

        # Each line is a JSON object
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            advisory = json.loads(line)
            advisories.append(
                {
                    "ghsaId": advisory.get("ghsa_id"),
                    "state": advisory.get("state"),
                    "summary": advisory.get("summary"),
                    "html_url": advisory.get("html_url"),
                    "cve_id": advisory.get("cve_id"),
                    "created_at": advisory.get("created_at"),
                    "updated_at": advisory.get("updated_at"),
                    "closed_at": advisory.get("closed_at"),
                }
            )
        break  # --paginate fetches all pages at once

    return advisories


def parse_covered_advisory_ids_from_file(file):
    """
    Parse advisory IDs from a stream of lines.
    Returns a set of covered advisory IDs.
    """
    covered_ids = set()
    first_line = re.compile(r"^\d+\..*$")
    second_line = re.compile(r"^(GHSA-\w{4}-\w{4}-\w{4}).*$")
    previous_line_is_first_line = False
    for line in file:
        line = line.strip()
        if first_line.match(line):
            previous_line_is_first_line = True
            continue
        if previous_line_is_first_line:
            # try and match as many GHSA IDs as possible, even on multiple lines,
            # as long as we don't see other text in between (apart from whitespace)
            # First split the line into words
            words = line.split()
            for word in words:
                # Check if the word matches the GHSA pattern
                if m := second_line.match(word):
                    covered_ids.add(m.group(1))
                else:
                    # If we find a word that doesn't match, we stop searching
                    # for GHSA IDs in this line or following lines.
                    previous_line_is_first_line = False
                    break
    return covered_ids


def get_covered_advisory_ids(filepath):
    """
    Opens the file and returns a set of covered advisory IDs.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return parse_covered_advisory_ids_from_file(f)
    except FileNotFoundError:
        print(
            f"Warning: Transparency report '{filepath}' not found. All advisories will be considered uncovered.",
            file=sys.stderr,
        )
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GitHub Security Advisory IDs for a repository using gh CLI."
    )
    parser.add_argument("--owner", required=True, help="Repository owner")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument(
        "--transparency-report",
        required=True,
        help="Path to text file listing covered advisories",
    )
    args = parser.parse_args()

    advisories = fetch_advisory_ids(args.owner, args.repo)
    covered_ids = get_covered_advisory_ids(args.transparency_report)

    # Sort advisories by created_at date (ascending), interpreting the string as a date
    # The date string is always present and has the format  2024-06-29T11:54:17Z
    # We can use strptime to convert it to a datetime object for sorting
    advisories.sort(
        key=lambda adv: (
            datetime.strptime(adv.get("created_at"), "%Y-%m-%dT%H:%M:%SZ")
            if adv.get("created_at")
            else datetime.max
        )
    )

    nr_reported = 0
    for advisory in advisories:
        covered = advisory["ghsaId"] in covered_ids
        if not covered and advisory["state"] in ("closed", "published"):
            # Only print uncovered advisories that are closed or published
            nr_reported += 1
            print(
                f"{nr_reported}. {advisory['ghsaId']} ({advisory['state']}): {advisory['summary']}\n"
                f"  URL: {advisory['html_url']}\n"
                f"  CVE: {advisory['cve_id']}\n"
                f"  Created: {advisory['created_at']}\n"
                f"  Updated: {advisory['updated_at']}\n"
                f"  Closed: {advisory['closed_at']}\n"
            )


import io
import unittest


class TestParseCoveredAdvisoryIdsFromFile(unittest.TestCase):
    def test_basic(self):
        content = """1. Some advisory title
    GHSA-aaaa-bbbb-cccc
2. Another advisory
    GHSA-dddd-eeee-ffff
"""
        file = io.StringIO(content)
        result = parse_covered_advisory_ids_from_file(file)
        self.assertEqual(result, {"GHSA-aaaa-bbbb-cccc", "GHSA-dddd-eeee-ffff"})

    def test_with_extra_lines(self):
        content = """
Some header
1. Advisory one
    GHSA-1111-2222-3333

Random text
2. Advisory two
    GHSA-4444-5555-6666
"""
        file = io.StringIO(content)
        result = parse_covered_advisory_ids_from_file(file)
        self.assertEqual(result, {"GHSA-1111-2222-3333", "GHSA-4444-5555-6666"})

    def test_no_advisories(self):
        content = "No advisories here\nJust text\n"
        file = io.StringIO(content)
        result = parse_covered_advisory_ids_from_file(file)
        self.assertEqual(result, set())

    def test_partial_match(self):
        content = """1. Title
    GHSA-xxxx-yyyy-zzzz
2. Title without id
    Not an advisory id
3. Another title
    GHSA-1234-5678-9abc
"""
        file = io.StringIO(content)
        result = parse_covered_advisory_ids_from_file(file)
        self.assertEqual(result, {"GHSA-xxxx-yyyy-zzzz", "GHSA-1234-5678-9abc"})

    def test_multiple_GHSAs(self):
        """
        Test that multiple GHSA IDs are returned, as long as there is no other text
        in between.
        Even when multiple GHSA IDs are on multiple lines, this should work.
        """
        content = """1. Some advisory title
    GHSA-aaaa-bbbb-cccc     GHSA-gggg-hhhh-iiii
    GHSA-dddd-eeee-ffff  other text GHSA-jjjj-kkkk-llll
    GHSA-mmmm-nnnn-oooo
"""
        file = io.StringIO(content)
        result = parse_covered_advisory_ids_from_file(file)
        self.assertEqual(
            result,
            {"GHSA-aaaa-bbbb-cccc", "GHSA-dddd-eeee-ffff", "GHSA-gggg-hhhh-iiii"},
        )


if __name__ == "__main__":
    # Run unittests if requested
    if "--test" in sys.argv:
        sys.argv.remove("--test")
        unittest.main()
        sys.exit(0)

    main()
