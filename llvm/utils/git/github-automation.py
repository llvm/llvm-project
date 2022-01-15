#!/usr/bin/env python3
#
# ======- github-automation - LLVM GitHub Automation Routines--*- python -*--==#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ==-------------------------------------------------------------------------==#

import argparse
import github
import os

class IssueSubscriber:

    @property
    def team_name(self) -> str:
        return self._team_name

    def __init__(self, token:str, repo:str, issue_number:int, label_name:str):
        self.repo = github.Github(token).get_repo(repo)
        self.org = github.Github(token).get_organization(self.repo.organization.login)
        self.issue = self.repo.get_issue(issue_number)
        self._team_name = 'issue-subscribers-{}'.format(label_name).lower()

    def run(self) -> bool:
        for team in self.org.get_teams():
            if self.team_name != team.name.lower():
                continue
            comment = '@llvm/{}'.format(team.slug)
            self.issue.create_comment(comment)
            return True
        return False


parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, required=True)
parser.add_argument('--repo', type=str, default=os.getenv('GITHUB_REPOSITORY', 'llvm/llvm-project'))
subparsers = parser.add_subparsers(dest='command')

issue_subscriber_parser = subparsers.add_parser('issue-subscriber')
issue_subscriber_parser.add_argument('--label-name', type=str, required=True)
issue_subscriber_parser.add_argument('--issue-number', type=int, required=True)

args = parser.parse_args()

if args.command == 'issue-subscriber':
    issue_subscriber = IssueSubscriber(args.token, args.repo, args.issue_number, args.label_name)
    issue_subscriber.run()
