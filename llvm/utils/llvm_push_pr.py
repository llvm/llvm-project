#!/usr/bin/env python3
"""A script to automate the creation and landing of a stack of Pull Requests."""

import argparse
import os
import re
import subprocess
import sys
import time
from typing import List, Optional

import requests


class Printer:
    """Handles all output and command execution, with options for dry runs and verbosity."""

    def __init__(
        self, dry_run: bool = False, verbose: bool = False, quiet: bool = False
    ):
        """Initializes the Printer with dry_run, verbose, and quiet settings."""
        self.dry_run = dry_run
        self.verbose = verbose
        self.quiet = quiet

    def print(self, message: str, file=sys.stdout):
        """Prints a message to the specified file, respecting quiet mode."""
        if self.quiet and file == sys.stdout:
            return
        print(message, file=file)

    def run_command(
        self,
        command: List[str],
        check: bool = True,
        capture_output: bool = False,
        text: bool = False,
        stdin_input: Optional[str] = None,
        read_only: bool = False,
    ) -> subprocess.CompletedProcess:
        """Runs a shell command, handling dry runs, verbosity, and errors."""
        if self.dry_run and not read_only:
            self.print(f"[Dry Run] Would run: {' '.join(command)}")
            return subprocess.CompletedProcess(command, 0, "", "")

        if self.verbose:
            self.print(f"Running: {' '.join(command)}")

        try:
            return subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=text,
                input=stdin_input,
            )
        except FileNotFoundError:
            self.print(
                f"Error: Command '{command[0]}' not found. Is it installed and in your PATH?",
                file=sys.stderr,
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            if check:
                self.print(
                    f"Error running command: {' '.join(command)}", file=sys.stderr
                )
                if e.stdout:
                    self.print(f"--- stdout ---\n{e.stdout}", file=sys.stderr)
                if e.stderr:
                    self.print(f"--- stderr ---\n{e.stderr}", file=sys.stderr)
                raise e
            return e


class GitHubAPI:
    """A wrapper for the GitHub API."""

    BASE_URL = "https://api.github.com"

    def __init__(self, repo_slug: str, printer: Printer, token: str):
        self.repo_slug = repo_slug
        self.printer = printer
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.BASE_URL}{endpoint}"
        if self.printer.verbose:
            self.printer.print(f"API Request: {method.upper()} {url}")
            if "json" in kwargs:
                self.printer.print(f"Payload: {kwargs['json']}")

        try:
            response = requests.request(
                method, url, headers=self.headers, timeout=30, **kwargs
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.printer.print(
                f"Error making API request to {url}: {e}", file=sys.stderr
            )
            if e.response is not None:
                self.printer.print(f"Response: {e.response.text}", file=sys.stderr)
            raise

    def get_user_login(self) -> str:
        """Gets the current user's login name."""
        response = self._request("get", "/user")
        return response.json()["login"]

    def create_pr(
        self,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> Optional[str]:
        """Creates a GitHub Pull Request."""
        self.printer.print(f"Creating pull request for '{head_branch}'...")
        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": draft,
        }
        response = self._request("post", f"/repos/{self.repo_slug}/pulls", json=data)
        pr_url = response.json().get("html_url")
        if not self.printer.dry_run:
            self.printer.print(f"Pull request created: {pr_url}")
        return pr_url

    def get_repo_settings(self) -> dict:
        """Gets repository settings."""
        response = self._request("get", f"/repos/{self.repo_slug}")
        return response.json()

    def merge_pr(self, pr_url: str):
        """Merges a PR, retrying if it's not yet mergeable."""
        if not pr_url:
            return

        if self.printer.dry_run:
            self.printer.print(f"[Dry Run] Would merge {pr_url}")
            return

        pr_number_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_number_match:
            self.printer.print(
                f"Could not extract PR number from URL: {pr_url}",
                file=sys.stderr,
            )
            sys.exit(1)
        pr_number = pr_number_match.group(1)

        head_branch = ""
        max_retries = 10
        retry_delay = 5  # seconds
        for i in range(max_retries):
            self.printer.print(
                f"Attempting to merge {pr_url} (attempt {i+1}/{max_retries})..."
            )

            pr_data_response = self._request(
                "get", f"/repos/{self.repo_slug}/pulls/{pr_number}"
            )
            pr_data = pr_data_response.json()
            head_branch = pr_data["head"]["ref"]

            if pr_data["mergeable"]:
                merge_data = {
                    "commit_title": f"{pr_data['title']} (#{pr_number})",
                    "merge_method": "squash",
                }
                try:
                    self._request(
                        "put",
                        f"/repos/{self.repo_slug}/pulls/{pr_number}/merge",
                        json=merge_data,
                    )
                    self.printer.print("Successfully merged.")
                    time.sleep(2)
                    return head_branch
                except requests.exceptions.RequestException as e:
                    if e.response and e.response.status_code == 405:
                        self.printer.print(
                            "PR not mergeable yet. Retrying in "
                            f"{retry_delay} seconds..."
                        )
                        time.sleep(retry_delay)
                    else:
                        raise e
            elif pr_data["mergeable_state"] == "dirty":
                self.printer.print("Error: Merge conflict.", file=sys.stderr)
                sys.exit(1)
            else:
                self.printer.print(
                    f"PR not mergeable yet ({pr_data['mergeable_state']}). "
                    f"Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)

        self.printer.print(
            f"Error: PR was not mergeable after {max_retries} attempts.",
            file=sys.stderr,
        )
        sys.exit(1)

    def enable_auto_merge(self, pr_url: str):
        """Enables auto-merge for a pull request."""
        if not pr_url:
            return

        if self.printer.dry_run:
            self.printer.print(f"[Dry Run] Would enable auto-merge for {pr_url}")
            return

        pr_number_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_number_match:
            self.printer.print(
                f"Could not extract PR number from URL: {pr_url}",
                file=sys.stderr,
            )
            sys.exit(1)
        pr_number = pr_number_match.group(1)

        self.printer.print(f"Enabling auto-merge for {pr_url}...")
        data = {
            "enabled": True,
            "merge_method": "squash",
        }
        self._request(
            "put",
            f"/repos/{self.repo_slug}/pulls/{pr_number}/auto-merge",
            json=data,
        )
        self.printer.print("Auto-merge enabled.")

    def delete_branch(self, branch_name: str, default_branch: Optional[str] = None):
        """Deletes a remote branch."""
        if default_branch and branch_name == default_branch:
            self.printer.print(
                f"Error: Refusing to delete the default branch '{branch_name}'.",
                file=sys.stderr,
            )
            return
        self.printer.print(f"Deleting remote branch '{branch_name}'")
        try:
            self._request(
                "delete", f"/repos/{self.repo_slug}/git/refs/heads/{branch_name}"
            )
        except requests.exceptions.RequestException as e:
            if (
                e.response is not None
                and e.response.status_code == 422
                and "Reference does not exist" in e.response.text
            ):
                if self.printer.verbose:
                    self.printer.print(
                        f"Warning: Remote branch '{branch_name}' was already deleted, skipping deletion.",
                        file=sys.stderr,
                    )
                return
            self.printer.print(
                f"Could not delete remote branch '{branch_name}': {e}",
                file=sys.stderr,
            )
            raise


class LLVMPRAutomator:
    """Automates the process of creating and landing a stack of GitHub Pull Requests."""

    def __init__(
        self,
        args: argparse.Namespace,
        printer: Printer,
        github_api: "GitHubAPI",
    ):
        self.args = args
        self.printer = printer
        self.github_api = github_api
        self.original_branch: str = ""
        self.repo_slug: str = ""
        self.created_branches: List[str] = []
        self.repo_settings: dict = {}

    def _run_cmd(self, command: List[str], read_only: bool = False, **kwargs):
        """Wrapper for run_command that passes the dry_run flag."""
        return self.printer.run_command(command, read_only=read_only, **kwargs)

    def _get_repo_slug(self) -> str:
        """Gets the GitHub repository slug from the remote URL."""
        result = self._run_cmd(
            ["git", "remote", "get-url", self.args.remote],
            capture_output=True,
            text=True,
            read_only=True,
        )
        url = result.stdout.strip()
        match = re.search(r"github\.com[/:]([\w.-]+/[\w.-]+)", url)
        if not match:
            self.printer.print(
                f"Error: Could not parse repository slug from remote URL: {url}",
                file=sys.stderr,
            )
            sys.exit(1)
        return match.group(1).replace(".git", "")

    def _get_current_branch(self) -> str:
        """Gets the current git branch."""
        result = self._run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        return result.stdout.strip()

    def _check_work_tree_is_clean(self):
        """Exits if the git work tree has uncommitted or unstaged changes."""
        result = self._run_cmd(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        if result.stdout.strip():
            self.printer.print(
                "Error: Your working tree is dirty. Please stash or commit your changes.",
                file=sys.stderr,
            )
            sys.exit(1)

    def _rebase_current_branch(self):
        """Rebases the current branch on top of the upstream base."""
        self._check_work_tree_is_clean()

        target = f"{self.args.upstream_remote}/{self.args.base}"
        self.printer.print(
            f"Fetching from '{self.args.upstream_remote}' and rebasing '{self.original_branch}' on top of '{target}'..."
        )
        self._run_cmd(["git", "fetch", self.args.upstream_remote, self.args.base])

        try:
            self._run_cmd(["git", "rebase", target])
        except subprocess.CalledProcessError as e:
            self.printer.print(
                "Error: The rebase operation failed, likely due to a merge conflict.",
                file=sys.stderr,
            )
            if e.stdout:
                self.printer.print(f"--- stdout ---\n{e.stdout}", file=sys.stderr)
            if e.stderr:
                self.printer.print(f"--- stderr ---\n{e.stderr}", file=sys.stderr)

            # Check if rebase is in progress before aborting
            rebase_status_result = self._run_cmd(
                ["git", "status", "--verify-status=REBASE_HEAD"],
                check=False,
                capture_output=True,
                text=True,
                read_only=True,
            )
            if (
                rebase_status_result.returncode == 0
            ):  # REBASE_HEAD exists, so rebase is in progress
                self.printer.print("Aborting rebase...", file=sys.stderr)
                self._run_cmd(["git", "rebase", "--abort"], check=False)
            sys.exit(1)

    def _get_commit_stack(self) -> List[str]:
        """Gets the stack of commits between the current branch's HEAD and its merge base with upstream."""
        target = f"{self.args.upstream_remote}/{self.args.base}"
        merge_base_result = self._run_cmd(
            ["git", "merge-base", "HEAD", target],
            capture_output=True,
            text=True,
            read_only=True,
        )
        merge_base = merge_base_result.stdout.strip()
        if not merge_base:
            self.printer.print(
                f"Error: Could not find a merge base between HEAD and {target}.",
                file=sys.stderr,
            )
            sys.exit(1)

        result = self._run_cmd(
            ["git", "rev-list", "--reverse", f"{merge_base}..HEAD"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        commits = result.stdout.strip().split("\n")
        return [c for c in commits if c]

    def _get_commit_details(self, commit_hash: str) -> tuple[str, str]:
        """Gets the title and body of a commit."""
        result = self._run_cmd(
            ["git", "show", "-s", "--format=%s%n%n%b", commit_hash],
            capture_output=True,
            text=True,
            read_only=True,
        )
        parts = result.stdout.strip().split("\n\n", 1)
        title = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        return title, body

    def _sanitize_for_branch_name(self, text: str) -> str:
        """Sanitizes a string to be used as a git branch name."""
        sanitized = re.sub(r"[^\w\s-]", "", text).strip().lower()
        sanitized = re.sub(r"[-\s]+", "-", sanitized)
        # Use "auto-pr" as a fallback.
        return sanitized or "auto-pr"

    def _create_and_push_branch_for_commit(
        self, commit_hash: str, base_branch_name: str, index: int
    ) -> str:
        """Creates and pushes a temporary branch pointing to a specific commit."""
        branch_name = f"{self.args.prefix}{base_branch_name}-{index + 1}"
        commit_title, _ = self._get_commit_details(commit_hash)
        self.printer.print(f"Processing commit {commit_hash[:7]}: {commit_title}")
        self.printer.print(f"Creating and pushing temporary branch '{branch_name}'")

        self._run_cmd(["git", "branch", "-f", branch_name, commit_hash])
        push_command = ["git", "push", self.args.remote, branch_name]
        self._run_cmd(push_command)
        self.created_branches.append(branch_name)
        return branch_name

    def run(self):
        """Main entry point for the automator, orchestrates the PR creation and merging process."""
        self.repo_slug = self._get_repo_slug()
        self.repo_settings = self.github_api.get_repo_settings()
        self.original_branch = self._get_current_branch()
        self.printer.print(f"On branch: {self.original_branch}")

        try:
            self._rebase_current_branch()
            initial_commits = self._get_commit_stack()

            if not initial_commits:
                self.printer.print("No new commits to process.")
                return

            if self.args.auto_merge and len(initial_commits) > 1:
                self.printer.print(
                    "Error: --auto-merge is only supported for a single commit.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if self.args.no_merge and len(initial_commits) > 1:
                self.printer.print(
                    "Error: --no-merge is only supported for a single commit. "
                    "For stacks, the script must merge sequentially.",
                    file=sys.stderr,
                )
                sys.exit(1)

            self.printer.print(f"Found {len(initial_commits)} commit(s) to process.")
            branch_base_name = self.original_branch
            if self.original_branch in ["main", "master"]:
                first_commit_title, _ = self._get_commit_details(initial_commits[0])
                branch_base_name = self._sanitize_for_branch_name(first_commit_title)

            for i in range(len(initial_commits)):
                if i > 0:
                    self._rebase_current_branch()

                commits = self._get_commit_stack()
                if not commits:
                    self.printer.print("Success! All commits have been landed.")
                    break

                commit_to_process = commits[0]
                commit_title, commit_body = self._get_commit_details(commit_to_process)

                temp_branch = self._create_and_push_branch_for_commit(
                    commit_to_process, branch_base_name, i
                )
                pr_url = self.github_api.create_pr(
                    head_branch=temp_branch,
                    base_branch=self.args.base,
                    title=commit_title,
                    body=commit_body,
                    draft=self.args.draft,
                )

                if not self.args.no_merge:
                    if self.args.auto_merge:
                        self.github_api.enable_auto_merge(pr_url)
                    else:
                        merged_branch = self.github_api.merge_pr(pr_url)
                        if merged_branch and not self.repo_settings.get(
                            "delete_branch_on_merge"
                        ):
                            self.github_api.delete_branch(
                                merged_branch, self.repo_settings.get("default_branch")
                            )

                    if temp_branch in self.created_branches:
                        self.created_branches.remove(temp_branch)

        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleans up by returning to the original branch and deleting all temporary branches."""
        self.printer.print(f"Returning to original branch: {self.original_branch}")
        self._run_cmd(["git", "checkout", self.original_branch], capture_output=True)
        if self.created_branches:
            self.printer.print("Cleaning up temporary local branches...")
            self._run_cmd(["git", "branch", "-D"] + self.created_branches)
            self.printer.print("Cleaning up temporary remote branches...")
            self._run_cmd(
                ["git", "push", self.args.remote, "--delete"] + self.created_branches,
                check=False,
            )


def check_prerequisites(printer: Printer):
    """Checks if git is installed and if inside a git repository."""
    printer.print("Checking prerequisites...")
    printer.run_command(["git", "--version"], capture_output=True, read_only=True)
    if not os.getenv("GITHUB_TOKEN"):
        printer.print(
            "Error: GITHUB_TOKEN environment variable not set.", file=sys.stderr
        )
        sys.exit(1)

    result = printer.run_command(
        ["git", "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
        read_only=True,
    )
    if result.returncode != 0 or result.stdout.strip() != "true":
        printer.print(
            "Error: This script must be run inside a git repository.", file=sys.stderr
        )
        sys.exit(1)
    printer.print("Prerequisites met.")


def main():
    """main entry point"""
    parser = argparse.ArgumentParser(
        description="Create and land a stack of Pull Requests."
    )
    GITHUB_REMOTE_NAME = "origin"
    UPSTREAM_REMOTE_NAME = "upstream"
    BASE_BRANCH = "main"

    printer = Printer()
    token = os.getenv("GITHUB_TOKEN")
    default_prefix = "dev/"
    if token:
        # Create a temporary API client to get the user login
        # We don't know the repo slug yet, so pass a dummy value.
        temp_api = GitHubAPI("", printer, token)
        try:
            user_login = temp_api.get_user_login()
            default_prefix = f"{user_login}/"
        except requests.exceptions.RequestException as e:
            printer.print(
                f"Could not fetch user login from GitHub: {e}", file=sys.stderr
            )

    parser.add_argument(
        "--base",
        default=BASE_BRANCH,
        help=f"Base branch to target (default: {BASE_BRANCH})",
    )
    parser.add_argument(
        "--remote",
        default=GITHUB_REMOTE_NAME,
        help=f"Remote for your fork to push to (default: {GITHUB_REMOTE_NAME})",
    )
    parser.add_argument(
        "--upstream-remote",
        default=UPSTREAM_REMOTE_NAME,
        help=f"Remote for the upstream repository (default: {UPSTREAM_REMOTE_NAME})",
    )
    parser.add_argument(
        "--prefix",
        default=default_prefix,
        help=f"Prefix for temporary branches (default: {default_prefix})",
    )
    parser.add_argument(
        "--draft", action="store_true", help="Create pull requests as drafts."
    )
    parser.add_argument(
        "--no-merge", action="store_true", help="Create PRs but do not merge them."
    )
    parser.add_argument(
        "--auto-merge",
        action="store_true",
        help="Enable auto-merge for each PR instead of attempting to merge immediately.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v", "--verbose", action="store_true", help="Print all commands being run."
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print only essential output and errors.",
    )

    args = parser.parse_args()
    if args.prefix and not args.prefix.endswith("/"):
        args.prefix += "/"

    printer = Printer(dry_run=args.dry_run, verbose=args.verbose, quiet=args.quiet)
    check_prerequisites(printer)

    # Get repo slug from git remote url
    result = printer.run_command(
        ["git", "remote", "get-url", args.remote],
        capture_output=True,
        text=True,
        read_only=True,
    )
    url = result.stdout.strip()
    match = re.search(r"github\.com[/:]([\w.-]+/[\w.-]+)", url)
    if not match:
        printer.print(
            f"Error: Could not parse repository slug from remote URL: {url}",
            file=sys.stderr,
        )
        sys.exit(1)
    repo_slug = match.group(1).replace(".git", "")

    github_api = GitHubAPI(repo_slug, printer, token)
    automator = LLVMPRAutomator(args, printer, github_api)
    automator.run()


if __name__ == "__main__":
    main()
