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
from git import Repo # type: ignore
import github
import os
import re
import requests
import sys
import time
from typing import *

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

def setup_llvmbot_git(git_dir = '.'):
    """
    Configure the git repo in `git_dir` with the llvmbot account so
    commits are attributed to llvmbot.
    """
    repo = Repo(git_dir)
    with repo.config_writer() as config:
        config.set_value('user', 'name', 'llvmbot')
        config.set_value('user', 'email', 'llvmbot@llvm.org')

def phab_api_call(phab_token:str, url:str, args:dict) -> dict:
    """
    Make an API call to the Phabricator web service and return a dictionary
    containing the json response.
    """
    data = { "api.token" : phab_token }
    data.update(args)
    response = requests.post(url, data = data)
    return response.json()


def phab_login_to_github_login(phab_token:str, repo:github.Repository.Repository, phab_login:str) -> str:
    """
    Tries to translate a Phabricator login to a github login by
    finding a commit made in Phabricator's Differential.
    The commit's SHA1 is then looked up in the github repo and
    the committer's login associated with that commit is returned.

    :param str phab_token: The Conduit API token to use for communication with Pabricator
    :param github.Repository.Repository repo: The github repo to use when looking for the SHA1 found in Differential
    :param str phab_login: The Phabricator login to be translated.
    """

    args = {
        "constraints[authors][0]" : phab_login,
        # PHID for "LLVM Github Monorepo" repository
        "constraints[repositories][0]" : "PHID-REPO-f4scjekhnkmh7qilxlcy",
        "limit" : 1
    }
    # API documentation: https://reviews.llvm.org/conduit/method/diffusion.commit.search/
    r = phab_api_call(phab_token, "https://reviews.llvm.org/api/diffusion.commit.search", args)
    data = r['result']['data']
    if len(data) == 0:
        # Can't find any commits associated with this user
        return None

    commit_sha = data[0]['fields']['identifier']
    committer = repo.get_commit(commit_sha).committer
    if not committer:
        # This committer had an email address GitHub could not recognize, so
        # it can't link the user to a GitHub account.
        print(f"Warning: Can't find github account for {phab_login}")
        return None
    return committer.login

def phab_get_commit_approvers(phab_token:str, repo:github.Repository.Repository, commit:github.Commit.Commit) -> list:
    args = { "corpus" : commit.commit.message }
    # API documentation: https://reviews.llvm.org/conduit/method/differential.parsecommitmessage/
    r = phab_api_call(phab_token, "https://reviews.llvm.org/api/differential.parsecommitmessage", args)
    review_id = r['result']['revisionIDFieldInfo']['value']
    if not review_id:
        # No Phabricator revision for this commit
        return []

    args = {
        'constraints[ids][0]' : review_id,
        'attachments[reviewers]' : True
    }
    # API documentation: https://reviews.llvm.org/conduit/method/differential.revision.search/
    r = phab_api_call(phab_token, "https://reviews.llvm.org/api/differential.revision.search", args)
    reviewers = r['result']['data'][0]['attachments']['reviewers']['reviewers']
    accepted = []
    for reviewer in reviewers:
        if reviewer['status'] != 'accepted':
            continue
        phid = reviewer['reviewerPHID']
        args = { 'constraints[phids][0]' : phid }
        # API documentation: https://reviews.llvm.org/conduit/method/user.search/
        r = phab_api_call(phab_token, "https://reviews.llvm.org/api/user.search", args)
        accepted.append(r['result']['data'][0]['fields']['username'])
    return accepted

class ReleaseWorkflow:

    CHERRY_PICK_FAILED_LABEL = 'release:cherry-pick-failed'

    """
    This class implements the sub-commands for the release-workflow command.
    The current sub-commands are:
        * create-branch
        * create-pull-request

    The execute_command method will automatically choose the correct sub-command
    based on the text in stdin.
    """

    def __init__(self, token:str, repo:str, issue_number:int,
                       branch_repo_name:str, branch_repo_token:str,
                       llvm_project_dir:str, phab_token:str) -> None:
        self._token = token
        self._repo_name = repo
        self._issue_number = issue_number
        self._branch_repo_name = branch_repo_name
        if branch_repo_token:
            self._branch_repo_token = branch_repo_token
        else:
            self._branch_repo_token = self.token
        self._llvm_project_dir = llvm_project_dir
        self._phab_token = phab_token

    @property
    def token(self) -> str:
        return self._token

    @property
    def repo_name(self) -> str:
        return self._repo_name

    @property
    def issue_number(self) -> int:
        return self._issue_number

    @property
    def branch_repo_name(self) -> str:
        return self._branch_repo_name

    @property
    def branch_repo_token(self) -> str:
        return self._branch_repo_token

    @property
    def llvm_project_dir(self) -> str:
        return self._llvm_project_dir

    @property
    def phab_token(self) -> str:
        return self._phab_token

    @property
    def repo(self) -> github.Repository.Repository:
        return github.Github(self.token).get_repo(self.repo_name)

    @property
    def issue(self) -> github.Issue.Issue:
        return self.repo.get_issue(self.issue_number)

    @property
    def push_url(self) -> str:
        return 'https://{}@github.com/{}'.format(self.branch_repo_token, self.branch_repo_name)

    @property
    def branch_name(self) -> str:
        return 'issue{}'.format(self.issue_number)

    @property
    def release_branch_for_issue(self) -> Optional[str]:
        issue = self.issue
        milestone = issue.milestone
        if milestone is None:
            return None
        m = re.search('branch: (.+)',milestone.description)
        if m:
            return m.group(1)
        return None

    def print_release_branch(self) -> None:
        print(self.release_branch_for_issue)

    def issue_notify_branch(self) -> None:
        self.issue.create_comment('/branch {}/{}'.format(self.branch_repo_name, self.branch_name))

    def issue_notify_pull_request(self, pull:github.PullRequest.PullRequest) -> None:
        self.issue.create_comment('/pull-request {}#{}'.format(self.branch_repo_name, pull.number))

    def make_ignore_comment(self, comment: str) -> str:
        """
        Returns the comment string with a prefix that will cause
        a Github workflow to skip parsing this comment.

        :param str comment: The comment to ignore
        """
        return "<!--IGNORE-->\n"+comment

    def issue_notify_no_milestone(self, comment:List[str]) -> None:
        message = "{}\n\nError: Command failed due to missing milestone.".format(''.join(['>' + line for line in comment]))
        self.issue.create_comment(self.make_ignore_comment(message))

    @property
    def action_url(self) -> str:
        if os.getenv('CI'):
            return 'https://github.com/{}/actions/runs/{}'.format(os.getenv('GITHUB_REPOSITORY'), os.getenv('GITHUB_RUN_ID'))
        return ""

    def issue_notify_cherry_pick_failure(self, commit:str) -> github.IssueComment.IssueComment:
        message = self.make_ignore_comment("Failed to cherry-pick: {}\n\n".format(commit))
        action_url = self.action_url
        if action_url:
            message += action_url + "\n\n"
        message += "Please manually backport the fix and push it to your github fork.  Once this is done, please add a comment like this:\n\n`/branch <user>/<repo>/<branch>`"
        issue = self.issue
        comment = issue.create_comment(message)
        issue.add_to_labels(self.CHERRY_PICK_FAILED_LABEL)
        return comment

    def issue_notify_pull_request_failure(self, branch:str) -> github.IssueComment.IssueComment:
        message = "Failed to create pull request for {} ".format(branch)
        message += self.action_url
        return self.issue.create_comment(message)

    def issue_remove_cherry_pick_failed_label(self):
        if self.CHERRY_PICK_FAILED_LABEL in [l.name for l in self.issue.labels]:
            self.issue.remove_from_labels(self.CHERRY_PICK_FAILED_LABEL)

    def pr_request_review(self, pr:github.PullRequest.PullRequest):
        """
        This function will try to find the best reviewers for `commits` and
        then add a comment requesting review of the backport and assign the
        pull request to the selected reviewers.

        The reviewers selected are those users who approved the patch in
        Phabricator.
        """
        reviewers = []
        for commit in pr.get_commits():
            approvers = phab_get_commit_approvers(self.phab_token, self.repo, commit)
            for a in approvers:
                login = phab_login_to_github_login(self.phab_token, self.repo, a)
                if not login:
                    continue
                reviewers.append(login)
        if len(reviewers):
            message = "{} What do you think about merging this PR to the release branch?".format(
                    " ".join(["@" + r for r in reviewers]))
            pr.create_issue_comment(message)
            pr.add_to_assignees(*reviewers)

    def create_branch(self, commits:List[str]) -> bool:
        """
        This function attempts to backport `commits` into the branch associated
        with `self.issue_number`.

        If this is successful, then the branch is pushed to `self.branch_repo_name`, if not,
        a comment is added to the issue saying that the cherry-pick failed.

        :param list commits: List of commits to cherry-pick.

        """
        print('cherry-picking', commits)
        branch_name = self.branch_name
        local_repo = Repo(self.llvm_project_dir)
        local_repo.git.checkout(self.release_branch_for_issue)

        for c in commits:
            try:
                local_repo.git.cherry_pick('-x', c)
            except Exception as e:
                self.issue_notify_cherry_pick_failure(c)
                raise e

        push_url = self.push_url
        print('Pushing to {} {}'.format(push_url, branch_name))
        local_repo.git.push(push_url, 'HEAD:{}'.format(branch_name), force=True)

        self.issue_notify_branch()
        self.issue_remove_cherry_pick_failed_label()
        return True

    def check_if_pull_request_exists(self, repo:github.Repository.Repository, head:str) -> bool:
        pulls = repo.get_pulls(head=head)
        return pulls.totalCount != 0

    def create_pull_request(self, owner:str, repo_name:str, branch:str) -> bool:
        """
        reate a pull request in `self.branch_repo_name`.  The base branch of the
        pull request will be chosen based on the the milestone attached to
        the issue represented by `self.issue_number`  For example if the milestone
        is Release 13.0.1, then the base branch will be release/13.x. `branch`
        will be used as the compare branch.
        https://docs.github.com/en/get-started/quickstart/github-glossary#base-branch
        https://docs.github.com/en/get-started/quickstart/github-glossary#compare-branch
        """
        repo = github.Github(self.token).get_repo(self.branch_repo_name)
        issue_ref = '{}#{}'.format(self.repo_name, self.issue_number)
        pull = None
        release_branch_for_issue = self.release_branch_for_issue
        if release_branch_for_issue is None:
            return False
        head_branch = branch
        if not repo.fork:
            # If the target repo is not a fork of llvm-project, we need to copy
            # the branch into the target repo.  GitHub only supports cross-repo pull
            # requests on forked repos.
            head_branch = f'{owner}-{branch}'
            local_repo = Repo(self.llvm_project_dir)
            push_done = False
            for i in range(0,5):
                try:
                    local_repo.git.fetch(f'https://github.com/{owner}/{repo_name}', f'{branch}:{branch}')
                    local_repo.git.push(self.push_url, f'{branch}:{head_branch}', force=True)
                    push_done = True
                    break
                except Exception as e:
                    print(e)
                    time.sleep(30)
                    continue
            if not push_done:
                raise Exception("Failed to mirror branch into {}".format(self.push_url))
            owner = repo.owner.login

        head = f"{owner}:{head_branch}"
        if self.check_if_pull_request_exists(repo, head):
            print("PR already exists...")
            return True
        try:
            pull = repo.create_pull(title=f"PR for {issue_ref}",
                                    body='resolves {}'.format(issue_ref),
                                    base=release_branch_for_issue,
                                    head=head,
                                    maintainer_can_modify=False)

            try:
                if self.phab_token:
                    self.pr_request_review(pull)
            except Exception as e:
                print("error: Failed while searching for reviewers", e)

        except Exception as e:
            self.issue_notify_pull_request_failure(branch)
            raise e

        if pull is None:
            return False

        self.issue_notify_pull_request(pull)
        self.issue_remove_cherry_pick_failed_label()

        # TODO(tstellar): Do you really want to always return True?
        return True


    def execute_command(self) -> bool:
        """
        This function reads lines from STDIN and executes the first command
        that it finds.  The 2 supported commands are:
        /cherry-pick commit0 <commit1> <commit2> <...>
        /branch <owner>/<repo>/<branch>
        """
        for line in sys.stdin:
            line.rstrip()
            m = re.search("/([a-z-]+)\s(.+)", line)
            if not m:
                continue
            command = m.group(1)
            args = m.group(2)

            if command == 'cherry-pick':
                return self.create_branch(args.split())

            if command == 'branch':
                m = re.match('([^/]+)/([^/]+)/(.+)', args)
                if m:
                    owner = m.group(1)
                    repo = m.group(2)
                    branch = m.group(3)
                    return self.create_pull_request(owner, repo, branch)

        print("Do not understand input:")
        print(sys.stdin.readlines())
        return False

parser = argparse.ArgumentParser()
parser.add_argument('--token', type=str, required=True, help='GitHub authentiation token')
parser.add_argument('--repo', type=str, default=os.getenv('GITHUB_REPOSITORY', 'llvm/llvm-project'),
                    help='The GitHub repository that we are working with in the form of <owner>/<repo> (e.g. llvm/llvm-project)')
subparsers = parser.add_subparsers(dest='command')

issue_subscriber_parser = subparsers.add_parser('issue-subscriber')
issue_subscriber_parser.add_argument('--label-name', type=str, required=True)
issue_subscriber_parser.add_argument('--issue-number', type=int, required=True)

release_workflow_parser = subparsers.add_parser('release-workflow')
release_workflow_parser.add_argument('--llvm-project-dir', type=str, default='.', help='directory containing the llvm-project checout')
release_workflow_parser.add_argument('--issue-number', type=int, required=True, help='The issue number to update')
release_workflow_parser.add_argument('--phab-token', type=str, help='Phabricator conduit API token. See https://reviews.llvm.org/settings/user/<USER>/page/apitokens/')
release_workflow_parser.add_argument('--branch-repo-token', type=str,
                                     help='GitHub authentication token to use for the repository where new branches will be pushed. Defaults to TOKEN.')
release_workflow_parser.add_argument('--branch-repo', type=str, default='llvm/llvm-project-release-prs',
                                     help='The name of the repo where new branches will be pushed (e.g. llvm/llvm-project)')
release_workflow_parser.add_argument('sub_command', type=str, choices=['print-release-branch', 'auto'],
                                     help='Print to stdout the name of the release branch ISSUE_NUMBER should be backported to')

llvmbot_git_config_parser = subparsers.add_parser('setup-llvmbot-git', help='Set the default user and email for the git repo in LLVM_PROJECT_DIR to llvmbot')

args = parser.parse_args()

if args.command == 'issue-subscriber':
    issue_subscriber = IssueSubscriber(args.token, args.repo, args.issue_number, args.label_name)
    issue_subscriber.run()
elif args.command == 'release-workflow':
    release_workflow = ReleaseWorkflow(args.token, args.repo, args.issue_number,
                                       args.branch_repo, args.branch_repo_token,
                                       args.llvm_project_dir, args.phab_token)
    if not release_workflow.release_branch_for_issue:
        release_workflow.issue_notify_no_milestone(sys.stdin.readlines())
        sys.exit(1)
    if args.sub_command == 'print-release-branch':
        release_workflow.print_release_branch()
    else:
        if not release_workflow.execute_command():
            sys.exit(1)
elif args.command == 'setup-llvmbot-git':
    setup_llvmbot_git()
