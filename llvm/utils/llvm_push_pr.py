#!/usr/bin/env python3
"""A script to automate the creation and landing of a stack of Pull Requests."""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from typing import List, Optional
from http.client import HTTPResponse
from dataclasses import dataclass

# --- Constants --- #
BASE_BRANCH = "main"
GITHUB_REMOTE_NAME = "origin"
UPSTREAM_REMOTE_NAME = "upstream"

LLVM_GITHUB_TOKEN_VAR = "LLVM_GITHUB_TOKEN"
LLVM_REPO = "llvm/llvm-project"
GITHUB_API = "https://api.github.com"

MERGE_MAX_RETRIES = 10
MERGE_RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 30  # seconds


class LlvmPrError(Exception):
    """Custom exception for errors in the PR automator script."""


@dataclass
class PRAutomatorConfig:
    """Configuration Data."""

    user_login: str
    token: str
    base_branch: str
    upstream_remote: str
    prefix: str
    draft: bool
    no_merge: bool
    auto_merge: bool


class CommandRunner:
    """Handles command execution and output.
    Supports dry runs and verbosity level."""

    def __init__(
        self, dry_run: bool = False, verbose: bool = False, quiet: bool = False
    ):
        self.dry_run = dry_run
        self.verbose = verbose
        self.quiet = quiet

    def print(self, message: str, file=sys.stdout) -> None:
        if self.quiet and file == sys.stdout:
            return
        print(message, file=file)

    def verbose_print(self, message: str, file=sys.stdout) -> None:
        if self.verbose:
            print(message, file)

    def run_command(
        self,
        command: List[str],
        check: bool = True,
        capture_output: bool = False,
        text: bool = False,
        stdin_input: Optional[str] = None,
        read_only: bool = False,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        if self.dry_run and not read_only:
            self.print(f"[Dry Run] Would run: {' '.join(command)}")
            return subprocess.CompletedProcess(command, 0, "", "")

        self.verbose_print(f"Running: {' '.join(command)}")

        try:
            return subprocess.run(
                command,
                check=check,
                capture_output=capture_output,
                text=text,
                input=stdin_input,
                env=env,
            )
        except FileNotFoundError as e:
            raise LlvmPrError(
                f"Command '{command[0]}' not found. Is it installed and in your PATH?"
            ) from e
        except subprocess.CalledProcessError as e:
            self.print(f"Error running command: {' '.join(command)}", file=sys.stderr)
            if e.stdout:
                self.print(f"--- stdout ---\n{e.stdout}", file=sys.stderr)
            if e.stderr:
                self.print(f"--- stderr ---\n{e.stderr}", file=sys.stderr)
            raise e


class GitHubAPI:
    """A wrapper for the GitHub API."""

    def __init__(self, runner: CommandRunner, token: str):
        self.runner = runner
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "llvm-push-pr",
        }
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPHandler(), urllib.request.HTTPSHandler()
        )

    def _request(
        self, method: str, endpoint: str, json_payload: Optional[dict] = None
    ) -> HTTPResponse:
        url = f"{GITHUB_API}{endpoint}"
        self.runner.verbose_print(f"API Request: {method.upper()} {url}")
        if json_payload:
            self.runner.verbose_print(f"Payload: {json_payload}")

        data = None
        headers = self.headers.copy()
        if json_payload:
            data = json.dumps(json_payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url, data=data, headers=headers, method=method.upper()
        )

        try:
            return self.opener.open(req, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as e:
            self.runner.print(
                f"Error making API request to {url}: {e}", file=sys.stderr
            )
            self.runner.verbose_print(
                f"Error response body: {e.read().decode()}", file=sys.stderr
            )
            raise e

    def _request_and_parse_json(
        self, method: str, endpoint: str, json_payload: Optional[dict] = None
    ) -> dict:
        with self._request(method, endpoint, json_payload) as response:
            # Expect a 200 'OK' or 201 'Created' status on success and JSON body.
            self._log_unexpected_status([200, 201], response.status)

            response_text = response.read().decode("utf-8")
            if response_text:
                return json.loads(response_text)
            return {}

    def _request_no_content(
        self, method: str, endpoint: str, json_payload: Optional[dict] = None
    ) -> None:
        with self._request(method, endpoint, json_payload) as response:
            # Expected a 204 No Content status on success, indicating the
            # operation was successful but there is no body.
            self._log_unexpected_status([204], response.status)

    def _log_unexpected_status(
        self, expected_statuses: List[int], actual_status: int
    ) -> None:
        if actual_status not in expected_statuses:
            self.runner.print(
                f"Warning: Expected status {', '.join(map(str, expected_statuses))}, but got {actual_status}",
                file=sys.stderr,
            )

    def get_user_login(self) -> str:
        return self._request_and_parse_json("GET", "/user")["login"]

    def create_pr(
        self,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
        draft: bool,
    ) -> Optional[str]:
        self.runner.print(f"Creating pull request for '{head_branch}'...")
        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
            "draft": draft,
        }
        response_data = self._request_and_parse_json(
            "POST", f"/repos/{LLVM_REPO}/pulls", json_payload=data
        )
        pr_url = response_data.get("html_url")
        if not self.runner.dry_run:
            self.runner.print(f"Pull request created: {pr_url}")
        return pr_url

    def get_repo_settings(self) -> dict:
        return self._request_and_parse_json("GET", f"/repos/{LLVM_REPO}")

    def _get_pr_details(self, pr_number: str) -> dict:
        """Fetches the JSON details for a given pull request number."""
        return self._request_and_parse_json(
            "GET", f"/repos/{LLVM_REPO}/pulls/{pr_number}"
        )

    def _attempt_squash_merge(self, pr_number: str) -> bool:
        """Attempts to squash merge a PR, returning True on success."""
        try:
            self._request_and_parse_json(
                "PUT",
                f"/repos/{LLVM_REPO}/pulls/{pr_number}/merge",
                json_payload={"merge_method": "squash"},
            )
            return True
        except urllib.error.HTTPError as e:
            # A 405 status code means the PR is not in a mergeable state.
            if e.code == 405:
                return False
            # Re-raise other HTTP errors.
            raise e

    def merge_pr(self, pr_url: str) -> Optional[str]:
        if not pr_url:
            return None

        if self.runner.dry_run:
            self.runner.print(f"[Dry Run] Would merge {pr_url}")
            return None

        pr_number_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_number_match:
            raise LlvmPrError(f"Could not extract PR number from URL: {pr_url}")
        pr_number = pr_number_match.group(1)

        for i in range(MERGE_MAX_RETRIES):
            self.runner.print(
                f"Attempting to merge {pr_url} (attempt {i + 1}/{MERGE_MAX_RETRIES})..."
            )

            pr_data = self._get_pr_details(pr_number)
            head_branch = pr_data["head"]["ref"]

            if pr_data.get("mergeable_state") == "dirty":
                raise LlvmPrError("Merge conflict.")

            if pr_data.get("mergeable"):
                if self._attempt_squash_merge(pr_number):
                    self.runner.print("Successfully merged.")
                    time.sleep(2)  # Allow GitHub's API to update.
                    return head_branch

            self.runner.print(
                f"PR not mergeable yet (state: {pr_data.get('mergeable_state', 'unknown')}). Retrying in {MERGE_RETRY_DELAY} seconds..."
            )
            time.sleep(MERGE_RETRY_DELAY)

        raise LlvmPrError(f"PR was not mergeable after {MERGE_MAX_RETRIES} attempts.")

    def enable_auto_merge(self, pr_url: str) -> None:
        if not pr_url:
            return

        if self.runner.dry_run:
            self.runner.print(f"[Dry Run] Would enable auto-merge for {pr_url}")
            return

        pr_number_match = re.search(r"/pull/(\d+)", pr_url)
        if not pr_number_match:
            raise LlvmPrError(f"Could not extract PR number from URL: {pr_url}")
        pr_number = pr_number_match.group(1)

        self.runner.print(f"Enabling auto-merge for {pr_url}...")
        data = {
            "enabled": True,
            "merge_method": "squash",
        }
        self._request_no_content(
            "PUT",
            f"/repos/{LLVM_REPO}/pulls/{pr_number}/auto-merge",
            json_payload=data,
        )
        self.runner.print("Auto-merge enabled.")

    def delete_branch(
        self, branch_name: str, default_branch: Optional[str] = None
    ) -> None:
        if default_branch and branch_name == default_branch:
            self.runner.print(
                f"Error: Refusing to delete the default branch '{branch_name}'.",
                file=sys.stderr,
            )
            return
        try:
            self._request_no_content(
                "DELETE", f"/repos/{LLVM_REPO}/git/refs/heads/{branch_name}"
            )
        except urllib.error.HTTPError as e:
            if e.code == 422:
                self.runner.print(
                    f"Warning: Remote branch '{branch_name}' was already deleted, skipping deletion.",
                    file=sys.stderr,
                )
            else:
                raise e


class LLVMPRAutomator:
    """Automates the process of creating and landing a stack of GitHub Pull Requests."""

    def __init__(
        self,
        runner: CommandRunner,
        github_api: "GitHubAPI",
        config: "PRAutomatorConfig",
        remote: str,
    ):
        self.runner = runner
        self.github_api = github_api
        self.config = config
        self.remote = remote
        self.original_branch: str = ""
        self.created_branches: List[str] = []
        self.repo_settings: dict = {}
        self._git_askpass_cmd = (
            f"python3 -c \"import os; print(os.environ['{LLVM_GITHUB_TOKEN_VAR}'])\""
        )

    def _run_cmd(
        self, command: List[str], read_only: bool = False, **kwargs
    ) -> subprocess.CompletedProcess:
        return self.runner.run_command(command, read_only=read_only, **kwargs)

    def _get_current_branch(self) -> str:
        result = self._run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        return result.stdout.strip()

    def _check_work_tree(self) -> None:
        result = self._run_cmd(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        if result.stdout.strip():
            raise LlvmPrError(
                "Your working tree is dirty. Please stash or commit your changes."
            )

    def _rebase_current_branch(self) -> None:
        self._check_work_tree()

        target = f"{self.config.upstream_remote}/{self.config.base_branch}"
        self.runner.print(
            f"Fetching from '{self.config.upstream_remote}' and rebasing '{self.original_branch}' on top of '{target}'..."
        )

        git_env = os.environ.copy()
        git_env["GIT_ASKPASS"] = self._git_askpass_cmd
        git_env[LLVM_GITHUB_TOKEN_VAR] = self.config.token
        git_env["GIT_TERMINAL_PROMPT"] = "0"

        https_upstream_url = self._get_https_url_for_remote(self.config.upstream_remote)
        refspec = f"refs/heads/{self.config.base_branch}:refs/remotes/{self.config.upstream_remote}/{self.config.base_branch}"
        self._run_cmd(["git", "fetch", https_upstream_url, refspec], env=git_env)

        try:
            self._run_cmd(["git", "rebase", target], env=git_env)
        except subprocess.CalledProcessError as e:
            self.runner.print(
                "Error: The rebase operation failed, likely due to a merge conflict.",
                file=sys.stderr,
            )
            if e.stdout:
                self.runner.print(f"--- stdout ---\n{e.stdout}", file=sys.stderr)
            if e.stderr:
                self.runner.print(f"--- stderr ---\n{e.stderr}", file=sys.stderr)

            # Check if rebase is in progress before aborting
            rebase_status_result = self._run_cmd(
                ["git", "status", "--verify-status=REBASE_HEAD"],
                check=False,
                capture_output=True,
                text=True,
                read_only=True,
                env=git_env,
            )

            # REBASE_HEAD exists, so rebase is in progress
            if rebase_status_result.returncode == 0:
                self.runner.print("Aborting rebase...", file=sys.stderr)
                self._run_cmd(["git", "rebase", "--abort"], check=False, env=git_env)
            raise LlvmPrError("rebase operation failed.") from e

    def _get_https_url_for_remote(self, remote_name: str) -> str:
        """Gets the URL for a remote and converts it to HTTPS if necessary."""
        remote_url_result = self._run_cmd(
            ["git", "remote", "get-url", remote_name],
            capture_output=True,
            text=True,
            read_only=True,
        )
        remote_url = remote_url_result.stdout.strip()
        if remote_url.startswith("git@github.com:"):
            return remote_url.replace("git@github.com:", "https://github.com/")
        if remote_url.startswith("https://github.com/"):
            return remote_url
        raise LlvmPrError(
            f"Unsupported remote URL format for {remote_name}: {remote_url}"
        )

    def _get_commit_stack(self) -> List[str]:
        target = f"{self.config.upstream_remote}/{self.config.base_branch}"
        merge_base_result = self._run_cmd(
            ["git", "merge-base", "HEAD", target],
            capture_output=True,
            text=True,
            read_only=True,
        )
        merge_base = merge_base_result.stdout.strip()
        if not merge_base:
            raise LlvmPrError(f"Could not find a merge base between HEAD and {target}.")

        result = self._run_cmd(
            ["git", "rev-list", "--reverse", f"{merge_base}..HEAD"],
            capture_output=True,
            text=True,
            read_only=True,
        )
        return result.stdout.strip().splitlines()

    def _get_commit_details(self, commit_hash: str) -> tuple[str, str]:
        # Get the subject and body from git show. Insert "\n\n" between to make
        # parsing simple to do w/ split.
        result = self._run_cmd(
            ["git", "show", "-s", "--format=%B", commit_hash],
            capture_output=True,
            text=True,
            read_only=True,
        )
        parts = [item.strip() for item in result.stdout.split("\n", 1)]
        title = parts[0]
        body = parts[1] if len(parts) > 1 else ""
        return title, body

    def _sanitize_branch_name(self, text: str) -> str:
        sanitized = re.sub(r"[^\w\s-]", "", text).strip().lower()
        sanitized = re.sub(r"[-\s]+", "-", sanitized)
        # Use "auto-pr" as a fallback.
        return sanitized or "auto-pr"

    def _validate_merge_config(self, num_commits: int) -> None:
        if num_commits > 1:
            if self.config.auto_merge:
                raise LlvmPrError("--auto-merge is only supported for a single commit.")

            if self.config.no_merge:
                raise LlvmPrError(
                    "--no-merge is only supported for a single commit. "
                    "For stacks, the script must merge sequentially."
                )

        self.runner.print(f"Found {num_commits} commit(s) to process.")

    def _create_and_push_branch_for_commit(
        self, commit_hash: str, base_branch_name: str, index: int
    ) -> str:
        branch_name = f"{self.config.prefix}{base_branch_name}-{index + 1}"
        commit_title, _ = self._get_commit_details(commit_hash)
        self.runner.print(f"Processing commit {commit_hash[:7]}: {commit_title}")
        self.runner.print(f"Pushing commit to temporary branch '{branch_name}'")

        git_env = os.environ.copy()
        git_env["GIT_ASKPASS"] = self._git_askpass_cmd
        git_env[LLVM_GITHUB_TOKEN_VAR] = self.config.token
        git_env["GIT_TERMINAL_PROMPT"] = "0"

        https_remote_url = self._get_https_url_for_remote(self.remote)

        push_command = [
            "git",
            "push",
            https_remote_url,
            f"{commit_hash}:refs/heads/{branch_name}",
        ]
        self._run_cmd(push_command, env=git_env)
        self.created_branches.append(branch_name)
        return branch_name

    def _process_commit(
        self, commit_hash: str, base_branch_name: str, index: int
    ) -> None:
        commit_title, commit_body = self._get_commit_details(commit_hash)

        temp_branch = self._create_and_push_branch_for_commit(
            commit_hash, base_branch_name, index
        )
        pr_url = self.github_api.create_pr(
            head_branch=f"{self.config.user_login}:{temp_branch}",
            base_branch=self.config.base_branch,
            title=commit_title,
            body=commit_body,
            draft=self.config.draft,
        )

        if self.config.no_merge:
            return

        if self.config.auto_merge:
            self.github_api.enable_auto_merge(pr_url)
        else:
            merged_branch = self.github_api.merge_pr(pr_url)
            if merged_branch and not self.repo_settings.get("delete_branch_on_merge"):
                # After a merge, the branch should be deleted.
                self.github_api.delete_branch(merged_branch)

        if temp_branch in self.created_branches:
            # If the branch was successfully merged, it should not be deleted
            # again during cleanup.
            self.created_branches.remove(temp_branch)

    def run(self) -> None:
        self.repo_settings = self.github_api.get_repo_settings()
        self.original_branch = self._get_current_branch()
        self.runner.print(f"On branch: {self.original_branch}")

        try:
            commits = self._get_commit_stack()
            if not commits:
                self.runner.print("No new commits to process.")
                return

            self._validate_merge_config(len(commits))
            branch_base_name = self.original_branch
            if self.original_branch == "main":
                first_commit_title, _ = self._get_commit_details(commits[0])
                branch_base_name = self._sanitize_branch_name(first_commit_title)

            for i in range(len(commits)):
                if not commits:
                    self.runner.print("Success! All commits have been landed.")
                    break
                self._process_commit(commits[0], branch_base_name, i)
                self._rebase_current_branch()
                # After a rebase, the commit hashes can change, so we need to
                # get the latest commit stack.
                commits = self._get_commit_stack()

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        self.runner.print(f"Returning to original branch: {self.original_branch}")
        self._run_cmd(["git", "checkout", self.original_branch], capture_output=True)
        if self.created_branches:
            self.runner.print("Cleaning up temporary remote branches...")
            for branch in self.created_branches:
                self.github_api.delete_branch(branch)


def check_prerequisites(runner: CommandRunner) -> None:
    runner.print("Checking prerequisites...")
    runner.run_command(["git", "--version"], capture_output=True, read_only=True)
    result = runner.run_command(
        ["git", "rev-parse", "--is-inside-work-tree"],
        check=False,
        capture_output=True,
        text=True,
        read_only=True,
    )

    if result.returncode != 0 or result.stdout.strip() != "true":
        raise LlvmPrError("This script must be run inside a git repository.")
    runner.print("Prerequisites met.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create and land a stack of Pull Requests."
    )

    command_runner = CommandRunner()
    token = os.getenv(LLVM_GITHUB_TOKEN_VAR)
    if not token:
        raise LlvmPrError(f"{LLVM_GITHUB_TOKEN_VAR} environment variable not set.")

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
        "--login",
        default=None,
        help="The user login to use. If not provided this will be queried from the TOKEN",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for temporary branches (default: users/<username>)",
    )
    parser.add_argument(
        "--draft", action="store_true", help="Create pull requests as drafts."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--no-merge", action="store_true", help="Create PRs but do not merge them."
    )
    group.add_argument(
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

    command_runner = CommandRunner(
        dry_run=args.dry_run, verbose=args.verbose, quiet=args.quiet
    )
    check_prerequisites(command_runner)

    github_api = GitHubAPI(command_runner, token)
    if not args.login:
        try:
            args.login = github_api.get_user_login()
        except urllib.error.HTTPError as e:
            raise LlvmPrError(f"Could not fetch user login from GitHub: {e}") from e

    if not args.prefix:
        args.prefix = f"users/{args.login}/"

    if not args.prefix.endswith("/"):
        args.prefix += "/"

    try:
        config = PRAutomatorConfig(
            user_login=args.login,
            token=token,
            base_branch=args.base,
            upstream_remote=args.upstream_remote,
            prefix=args.prefix,
            draft=args.draft,
            no_merge=args.no_merge,
            auto_merge=args.auto_merge,
        )
        automator = LLVMPRAutomator(
            runner=command_runner,
            github_api=github_api,
            config=config,
            remote=args.remote,
        )
        automator.run()
    except LlvmPrError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
