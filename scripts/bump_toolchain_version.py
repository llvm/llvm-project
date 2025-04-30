import logging
import re
from pathlib import Path
import os
import argparse
from modules.git.git import NextGit


update_types = ["MAJOR", "MINOR", "PATCHLEVEL"]


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-type",
        required=True,
        type=str,
        default="",
        choices=update_types,
        help="Set version update type",
    )
    parser.add_argument("--commit-msg", type=str, default="", help="Git commit message")
    parser.add_argument("--force", action="store_true", help="Force push commit")
    parser.add_argument(
        "--toolchain-path",
        required=True,
        type=str,
        default="",
        help="Next llvm project source code path",
    )
    parser.add_argument(
        "--utils-path",
        type=str,
        default="",
        help="Nextutils source code path",
    )
    parser.add_argument(
        "--toolchain-sha", type=str, default="", help="Reset to toolchain version sha"
    )
    parser.add_argument(
        "--utils-sha", type=str, default="", help="Reset to toolchain version sha"
    )
    parser.add_argument("--bump", action="store_true", help="Bump toolchain version")
    parser.add_argument("--tag", action="store_true", help="Tag toolchain version")
    parser.add_argument(
        "--push", action="store_true", help="Push toolchain version into origin repo"
    )
    parser.add_argument(
        "--branch", type=str, default="", help="Bump toolchain version branch name"
    )
    return parser


class bumpToolchainGit(NextGit):
    """
    Bump toolchain version utility
    """

    def __init__(self, user_vars):
        super().__init__()
        self.user_vars = user_vars
        self.repo_list = []
        self.validate_vars()
        self.repo_data = self.get_repos_data()
        self.validate_repos()

    def validate_vars(self):
        """Validate user vars"""

        if (
            not self.user_vars.bump
            and not self.user_vars.toolchain_sha
            and not self.user_vars.utils_sha
        ):
            raise ValueError(
                "Error: argument --bump or --toolchain-sha/--utils-sha required"
            )

        if self.user_vars.bump and (
            self.user_vars.toolchain_sha or self.user_vars.utils_sha
        ):
            raise ValueError(
                "Error: argument --bump not allowed with arguments --toolchain-sha/--utils-sha"
            )

    def get_repos_data(self):
        """Return toolchain/nextutils repo data"""

        repo_data = {
            "toolchain": {
                "version_file": Path("nextsilicon/nextcc/CMakeLists.txt"),
                "text": "NEXT_TOOLCHAIN_VERSION",
                "sha": self.user_vars.toolchain_sha,
                "path": Path(self.user_vars.toolchain_path)
                if self.user_vars.toolchain_path
                else None,
            },
            "nextutils": {
                "version_file": Path("cmake/NextLLVMVersion.cmake"),
                "text": "NEXT_TOOLCHAIN_EXPECTED",
                "sha": self.user_vars.utils_sha,
                "path": Path(self.user_vars.utils_path)
                if self.user_vars.utils_path
                else None,
            },
        }

        for update_type in update_types:
            for repo in repo_data:
                repo_data[repo][update_type] = None

        return repo_data

    def validate_repos(self):
        """Validate toolchain/nextutils repo path"""

        for repo, repo_data in self.repo_data.items():
            if not repo_data["path"]:
                continue
            if not repo_data["path"].is_dir():
                raise OSError(f"Missing {repo} path: {repo_data['path'].as_posix()}")
            if not repo_data["path"].joinpath(repo_data["version_file"]).is_file():
                raise OSError(
                    f"Missing {repo} version file: {repo_data['version_file'].as_posix()}"
                )
            self.repo_list.append(repo)

        if not self.repo_list:
            raise ValueError(
                "Missing valid repo path, please enter --toolchain-path/--utils-path"
            )

    def reset_version_commit(self):
        """reset to toolchain version commit"""

        for repo in self.repo_list:
            os.chdir(self.repo_data[repo]["path"])
            if not self.repo_data[repo]["sha"]:
                continue
            self.logger.info(
                f"Reset to toolchain commit: {self.repo_data[repo]['sha']}"
            )
            if not self.git_reset(self.repo_data[repo]["sha"], True):
                raise OSError(
                    f"Failed to reset {repo} to {self.repo_data[repo]['sha']}"
                )

    def get_branch(self, path):
        """get toolchain/nextutils repo current git branch"""

        os.chdir(path)
        return self.get_current_branch()

    def checkout_branch(self, path, branch, create=False, validate=False):
        os.chdir(path)
        if validate and self.git_validate_branch(branch, True):
            if not self.git_delete_branch(branch, True):
                return False
        return self.git_checkout(branch, create)

    def get_version_text(self, repo, update_type):
        return f"{self.repo_data[repo]['text']}_{update_type}"

    def get_version_file(self, repo):
        return self.repo_data[repo]["path"].joinpath(
            self.repo_data[repo]["version_file"]
        )

    def set_branches(self):
        """set toolchain/nextutils repo data branch"""

        for repo in self.repo_list:
            if self.repo_data[repo].get("branch"):
                continue

            if self.user_vars.branch:
                if not self.checkout_branch(
                    self.repo_data[repo]["path"], self.user_vars.branch, True, True
                ):
                    raise OSError(
                        f"Failed to create {repo} {self.user_vars.branch} branch"
                    )
                self.repo_data[repo]["branch"] = self.user_vars.branch
                continue

            self.repo_data[repo]["branch"] = self.get_branch(
                self.repo_data[repo]["path"]
            )

            if not self.repo_data[repo]["branch"]:
                raise OSError(f"Failed to validate {repo} branch")

            self.logger.info(f"{repo} branch set to {self.repo_data[repo]['branch']}")

    def get_version_file_lines(self, repo):
        """return version file lines"""

        os.chdir(self.repo_data[repo]["path"])

        with open(self.get_version_file(repo), "r") as f:
            lines = f.readlines()

        return lines

    def find_version(self, repo, text, lines, target_version=None):
        """find toolchain version text with optional version update"""

        version = None
        regex_text = rf"{text} (\d+)"
        regex = re.compile(regex_text)

        for idx, line in enumerate(lines):
            regex_found = regex.search(line)
            if not regex_found:
                continue
            version = regex_found[1]
            if not target_version:
                break
            lines[idx] = lines[idx].replace(
                f"{text} {version}", f"{text} {target_version}"
            )
            break

        if version is None:
            raise Exception(
                f"Failed to find {text} in {self.repo_data[repo]['path'].joinpath(self.repo_data[repo]['version_file']).as_posix()}"
            )

        return version, lines

    def set_repos_version(self):
        """set repos version data"""

        for repo in self.repo_list:
            lines = self.get_version_file_lines(repo)
            for update_type in update_types:
                self.repo_data[repo][update_type], _ = self.find_version(
                    repo,
                    self.get_version_text(repo, update_type),
                    lines,
                    False,
                )

    def bump_repos_version(self):
        """bump repos version data"""

        for repo in self.repo_list:
            if repo == "toolchain":
                if self.user_vars.update_type == "MAJOR":
                    self.repo_data[repo]["MAJOR"] = str(
                        int(self.repo_data[repo]["MAJOR"]) + 1
                    )
                    self.repo_data[repo]["MINOR"] = "0"
                    self.repo_data[repo]["PATCHLEVEL"] = "0"
                elif self.user_vars.update_type == "MINOR":
                    self.repo_data[repo]["MINOR"] = str(
                        int(self.repo_data[repo]["MINOR"]) + 1
                    )
                    self.repo_data[repo]["PATCHLEVEL"] = "0"
                else:
                    self.repo_data[repo]["PATCHLEVEL"] = str(
                        int(self.repo_data[repo]["PATCHLEVEL"]) + 1
                    )
            else:
                for update_type in update_types:
                    self.repo_data[repo][update_type] = self.repo_data["toolchain"][
                        update_type
                    ]

    def get_full_version(self, repo):
        """return full version string"""

        version = []

        for update_type in update_types:
            version.append(self.repo_data[repo][update_type])

        return ".".join(version)

    def update_version_file(self, repo):
        """update version file"""

        lines = self.get_version_file_lines(repo)

        for update_type in update_types:
            _, lines = self.find_version(
                repo,
                self.get_version_text(repo, update_type),
                lines,
                self.repo_data[repo][update_type],
            )

        with open(self.get_version_file(repo), "w") as f:
            f.write("".join(lines))

    def update_version_files(self):
        """update version files"""

        for repo in self.repo_list:
            self.logger.info(
                f"Updating {repo} file {self.repo_data[repo]['path'].joinpath(self.repo_data[repo]['version_file'])}"
            )
            self.update_version_file(repo)
            self.logger.info(
                f"Updating {repo} {self.user_vars.update_type.lower()} version to {self.repo_data[repo][self.user_vars.update_type]}"
            )

    def commit_version_file(self, repo):
        """commit version file"""

        os.chdir(self.repo_data[repo]["path"])
        commit_msg = self.user_vars.commit_msg

        if not commit_msg:
            commit_msg = f"next toolchain: bump toolchain {self.user_vars.update_type.lower()} version to {self.get_full_version(repo)}"

        if not self.git_add([self.repo_data[repo]["version_file"].as_posix()]):
            return

        if not self.git_commit(commit_msg):
            return

        return True

    def tag_repo_version(self, repo):
        """tag repo toolchain version"""

        os.chdir(self.repo_data[repo]["path"])

        if self.git_tag(f"toolchain-{self.get_full_version(repo)}"):
            return

        raise Exception(f"Failed to tag {repo} repo toolchain version")

    def push_repo_version(self, repo):
        """push repo version into origin"""

        os.chdir(self.repo_data[repo]["path"])
        if self.git_push(
            self.repo_data[repo]["branch"], self.user_vars.force, self.user_vars.tag
        ):
            return

        raise Exception(f"Failed to push toolchain version into {repo} repo")

    def bump_toolchain_version(self):
        """bump toolchain version"""

        self.logger.info(
            f"Bump {self.user_vars.update_type.lower()} toolchain version for: {', '.join(self.repo_list)}"
        )

        self.set_repos_version()
        self.bump_repos_version()
        self.update_version_files()

    def commit_toolchain_version(self):
        """commit toolchain/nextutils version files"""

        self.logger.info(
            f"Commit {self.user_vars.update_type.lower()} toolchain version for: {', '.join(self.repo_list)}"
        )

        for repo in self.repo_list:
            if self.commit_version_file(repo):
                continue
            raise OSError(f"Failed to commit {repo}")

    def tag_toolchain_version(self):
        """tag toolchain version"""

        self.logger.info(
            f"Tag {self.user_vars.update_type.lower()} toolchain version for: {', '.join(self.repo_list)}"
        )

        self.set_repos_version()

        for repo in self.repo_list:
            self.tag_repo_version(repo)

    def push_toolchain_version(self):
        """push toolchain version into origin toolchain/nextutils repo"""

        self.logger.info(
            f"Push {self.user_vars.update_type.lower()} toolchain version for: {', '.join(self.repo_list)}"
        )

        for repo in self.repo_list:
            self.push_repo_version(repo)


def main():
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
    parser = build_arg_parser()
    user_vars = parser.parse_args()
    bump_toolchain_git = bumpToolchainGit(user_vars)
    bump_toolchain_git.set_branches()

    if user_vars.toolchain_sha or user_vars.utils_sha:
        bump_toolchain_git.reset_version_commit()
    else:
        bump_toolchain_git.bump_toolchain_version()
        bump_toolchain_git.commit_toolchain_version()

    if user_vars.tag:
        bump_toolchain_git.tag_toolchain_version()

    if user_vars.push:
        bump_toolchain_git.push_toolchain_version()


if __name__ == "__main__":
    main()
