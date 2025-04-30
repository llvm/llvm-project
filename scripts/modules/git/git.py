import sys
import logging
import subprocess


class NextGit:
    """
    Base git commands executer
    """

    def __init__(self):
        self.logger = logging

    def get_current_branch(self):
        """get git current branch"""

        cmd = ["git", "symbolic-ref", "--short", "HEAD"]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to get git branch")
            return ""

        branch = res.stdout.strip()
        if branch == "detached":
            self.logger.error("Failed to get branch, detached")
            return ""

        return branch

    def get_branch_sha(self, branch, local=False):
        """get git branch sha"""

        ref = f"origin/{branch}"

        if local:
            ref = f"refs/heads/{branch}"

        cmd = ["git", "rev-parse", ref]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to get git branch sha")
            return ""

        return res.stdout.strip()

    def git_reset(self, sha, hard=False):
        """git reset"""

        cmd = ["git", "reset"]

        if hard:
            cmd.append("--hard")

        cmd.append(sha)

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to reset git")
            return

        return True

    def git_checkout(self, branch, create_branch=False, source=None):
        """git checkout branch"""

        cmd = ["git", "checkout"]

        if create_branch:
            cmd.append("-b")

        cmd.append(branch)

        if source:
            cmd.append(source)

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to checkout git branch")
            return

        return True

    def git_validate_branch(self, branch, local=False):
        """git validate if branch exists"""

        ref = f"origin/{branch}"

        if local:
            ref = f"refs/heads/{branch}"

        cmd = ["git", "show-ref", "--quiet", ref]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to validate git branch")
            return False

        if res.returncode != 0:
            return False

        return True

    def git_delete_branch(self, branch, force=False):
        """git delete branch"""

        cmd = ["git", "branch", "-d"]

        if force:
            cmd.append("--force")

        cmd.append(branch)

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to delete git branch")
            return

        return True

    def git_diff(self):
        """git diff"""

        cmd = ["git", "diff"]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to get git diff")
            return

        return res.stdout

    def git_status(self):
        """git status"""

        cmd = ["git", "status"]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to get git status")
            return

        return res.stdout

    def git_show(self, sha="HEAD", format=""):
        """git show"""

        cmd = ["git", "show", "-s", sha]

        if format:
            cmd.append(format)

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to get git show")
            return

        return res.stdout

    def get_commit_author(self, sha="HEAD"):
        """get git commit author"""

        return self.git_show(sha, "--format=%an").strip()

    def get_commit_msg(self, sha="HEAD"):
        """get git commit message"""

        return self.git_show(sha, "--format=%s").strip()

    def git_add(self, updated_files="."):
        """git add changes"""

        cmd = ["git", "add"] + updated_files

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to add git changes")
            return

        return True

    def git_commit(self, commit_msg):
        """git commit changes"""

        cmd = ["git", "commit", "-s", "-m", f'"{commit_msg}"']

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to commit git changes")
            return

        return True

    def git_tag(self, tag):
        """git tag"""

        cmd = ["git", "tag", tag]

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to tag git")
            return

        return True

    def git_push(self, branch, force=False, tags=False):
        """git push changes to origin"""

        cmd = ["git", "push", "origin", branch]

        if force:
            cmd.append("--force")

        if tags:
            cmd.append("--tags")

        res = self.command(cmd)

        if not res:
            self.logger.error("Failed to push git changes")
            return

        self.logger.info(f"Changes were pushed into origin branch {branch}")

        return True

    def command(self, cmd):
        """execute command"""

        cmd = " ".join(cmd)

        self.logger.info(f"Command: {cmd}")

        try:
            res = subprocess.run(
                cmd, capture_output=True, check=False, text=True, shell=True
            )
        except subprocess.SubprocessError:
            self.logger.error(f"{sys.exc_info()[0]}: {sys.exc_info()[1]}")
            return

        if not res or res.returncode != 0:
            self.logger.error(f"Failed to run command: {cmd}")
            return

        return res
