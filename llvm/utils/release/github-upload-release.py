#!/usr/bin/env python3
# ===-- github-upload-release.py  ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
#
# Create and manage releases in the llvm github project.
#
# This script requires python3 and the PyGithub module.
#
# Example Usage:
#
# You will need to obtain a personal access token for your github account in
# order to use this script.  Instructions for doing this can be found here:
# https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line
#
# Create a new release from an existing tag:
# ./github-upload-release.py --token $github_token --release 8.0.1-rc4 create
#
# Upload files for a release
# ./github-upload-release.py --token $github_token --release 8.0.1-rc4 upload --files llvm-8.0.1rc4.src.tar.xz
#
# You can upload as many files as you want at a time and use wildcards e.g.
# ./github-upload-release.py --token $github_token --release 8.0.1-rc4 upload --files *.src.*
# ===------------------------------------------------------------------------===#


import argparse
import github
import sys
from textwrap import dedent


def create_release(repo, release, tag=None, name=None, message=None):
    if not tag:
        tag = "llvmorg-{}".format(release)

    if not name:
        name = "LLVM {}".format(release)

    if not message:
        # Note that these lines are not length limited because if we do so, GitHub
        # assumes that should be how it is laid out on the page. We want GitHub to
        # do the reflowing for us instead.
        #
        # Once all the atuomatic binary builds have completed, the HTML comments
        # with UPPERCASE markers in them will be removed to reveal the download
        # links later. Other lines are surrounded in <!-- --> for release uploaders
        # to manually uncomment when they upload that package.
        message = dedent(
            """\
## LLVM {release} Release

<!-- AUTOMATIC_DOWNLOAD_LINKS_BEGIN
* [Linux x86_64](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-Linux-X64.tar.xz) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-Linux-X64.tar.xz.jsonl))
* [Linux Arm64](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-Linux-ARM64.tar.xz) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-Linux-ARM64.tar.xz.jsonl))
AUTOMATIC_DOWNLOAD_LINKS_END -->
<!-- * [Linux Armv7-a](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-armv7a-linux-gnueabihf.tar.gz) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-armv7a-linux-gnueabihf.tar.gz.sig)) -->

<!-- AUTOMATIC_DOWNLOAD_LINKS_BEGIN
* [macOS Apple Silicon](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-macOS-ARM64.tar.xz) (ARM64) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-macOS-ARM64.tar.xz.jsonl))
* [macOS Intel](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-macOS-X64.tar.xz) (x86-64) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-macOS-X64.tar.xz.jsonl))
AUTOMATIC_DOWNLOAD_LINKS_END -->

<!-- * Windows x64 (64-bit): [installer](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-win64.exe) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-win64.exe.sig)), [archive](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-x86_64-pc-windows-msvc.tar.xz) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-x86_64-pc-windows-msvc.tar.xz.sig)) -->
<!-- * Windows x86 (32-bit): [installer](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-win32.exe) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-win32.exe.sig)) -->
<!-- * Windows on Arm (ARM64): [installer](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-woa64.exe) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/LLVM-{release}-woa64.exe.sig)), [archive](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-aarch64-pc-windows-msvc.tar.xz) ([signature](https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/clang+llvm-{release}-aarch64-pc-windows-msvc.tar.xz.sig)) -->

Download links will appear here once builds have completed. <!-- AUTOMATIC_DOWNLOAD_LINKS_PLACEHOLDER -->

For any other variants of platform and architecture, check the full list of release packages at the bottom of this release page. If you do not find a release package for your platform, you may be able to find a community built package on the LLVM Discourse forum thread for this release. Remember that these are built by volunteers and may not always be available. If you rely on a platform or configuration that is not one of the defaults, we suggest you use the binaries that your platform provides, or build your own release packages.

## Package Types

Each platform has one binary release package. The file name starts with either `LLVM-` or `clang+llvm-` and ends with the platform's name. For example, `LLVM-{release}-Linux-ARM64.tar.xz` contains LLVM binaries for Arm64 Linux.

Except for Windows. Where `LLVM-*.exe` is an installer intended for using LLVM as a toolchain and the archive `clang+llvm-` contains the contents of the installer, plus libraries and tools not normally used in a toolchain. You most likely want the `LLVM-` installer, unless you are developing software which itself uses LLVM, in which case choose `clang+llvm-`.

In addition, source archives are available:
* `<sub-project>-{release}.src.tar.xz` are archives of the sources of specific sub-projects of `llvm-project` (except for `test-suite` which is an archive of the [LLVM Test Suite](https://github.com/llvm/llvm-test-suite)).
* To get all the `llvm-project` source code for this release, choose `llvm-project-{release}.src.tar.xz`.

## Verifying Packages

All packages come with a matching `.sig` or `.jsonl` file. You should use these to verify the integrity of the packages.

If it has a `.sig` file, it should have been signed by the release managers using GPG. Download the keys from the [LLVM website](https://releases.llvm.org/release-keys.asc), import them into your keyring and use them to verify the file:
```
$ gpg --import release-keys.asc
$ gpg --verify <package file name>.sig <package file name>
```

If it has a `.jsonl` file, use [gh](https://cli.github.com/manual/gh_attestation_verify) to verify the package:
```
$ gh attestation verify --repo llvm/llvm-project <package file name>
(if you are able to connect to GitHub)
$ gh attestation verify --repo llvm/llvm-project <package file name> --bundle <package file name>.jsonl
(using attestation file on disk)
```"""
        ).format(release=release)

    prerelease = True if "rc" in release else False

    repo.create_git_release(tag=tag, name=name, message=message, prerelease=prerelease)


def upload_files(repo, release, files):
    release = repo.get_release("llvmorg-{}".format(release))
    for f in files:
        print("Uploading {}".format(f))
        release.upload_asset(f)
        print("Done")


def uncomment_download_links(repo, release):
    release = repo.get_release("llvmorg-{}".format(release))

    new_message = []
    to_remove = [
        "AUTOMATIC_DOWNLOAD_LINKS_BEGIN",
        "AUTOMATIC_DOWNLOAD_LINKS_END",
        "AUTOMATIC_DOWNLOAD_LINKS_PLACEHOLDER",
    ]
    for line in release.body.splitlines():
        for comment in to_remove:
            if comment in line:
                break
        else:
            new_message.append(line)

    release.update_release(
        name=release.title,
        message="\n".join(new_message),
        draft=release.draft,
        prerelease=release.prerelease,
    )


parser = argparse.ArgumentParser()
parser.add_argument(
    "command",
    type=str,
    choices=["create", "upload", "check-permissions", "uncomment_download_links"],
)

# All args
parser.add_argument("--token", type=str)
parser.add_argument("--release", type=str)
parser.add_argument("--user", type=str)
parser.add_argument("--user-token", type=str)

# Upload args
parser.add_argument("--files", nargs="+", type=str)

args = parser.parse_args()

gh = github.Github(args.token)
llvm_org = gh.get_organization("llvm")
llvm_repo = llvm_org.get_repo("llvm-project")

if args.user:
    if not args.user_token:
        print("--user-token option required when --user is used")
        sys.exit(1)
    # Validate that this user is allowed to modify releases.
    user = gh.get_user(args.user)
    team = (
        github.Github(args.user_token)
        .get_organization("llvm")
        .get_team_by_slug("llvm-release-managers")
    )
    if not team.has_in_members(user):
        print("User {} is not a allowed to modify releases".format(args.user))
        sys.exit(1)
elif args.command == "check-permissions":
    print("--user option required for check-permissions")
    sys.exit(1)

if args.command == "create":
    create_release(llvm_repo, args.release)
if args.command == "upload":
    upload_files(llvm_repo, args.release, args.files)
if args.command == "uncomment_download_links":
    uncomment_download_links(llvm_repo, args.release)
