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

# Requirements and lockfiles can be found in llvm/utils/git/requirements.upload_release.txt.in
# and llvm/utils/git/requirements.upload_release.txt
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


# This is a tuple of sections of download links. Each section then contains
# an entry for each line in that section. Each line entry contains:
# * A unique tag to go in "<!-- <tag>" so we can find it later in the release
#   message.
# * A format string for the line's content.
# * A list of filenames to substitute into the format string. Before substitution into
#   the format string, file names will have the base download URL prepended to
#   them and 'release' replaced with the release version.
#
# Between each set of links, an empty line will be added.
#
# This data is used to generate the links for the release message, and by
# uncomment_download_links to verify whether for any given link line, all files
# linked to are present in the release's assets.
release_links = (
    (
        (
            "LINUX_X86",
            "* [Linux x86_64]({0}) ([signature]({1}))",
            (
                "LLVM-{release}-Linux-X64.tar.xz",
                "LLVM-{release}-Linux-X64.tar.xz.jsonl",
            ),
        ),
        (
            "LINUX_ARM64",
            "* [Linux Arm64]({0}) ([signature]({1}))",
            (
                "LLVM-{release}-Linux-ARM64.tar.xz",
                "LLVM-{release}-Linux-ARM64.tar.xz.jsonl",
            ),
        ),
        (
            "LINUX_ARMV7A",
            "* [Linux Armv7-a]({0}) ([signature]({1}))",
            (
                "clang+llvm-{release}-armv7a-linux-gnueabihf.tar.gz",
                "clang+llvm-{release}-armv7a-linux-gnueabihf.tar.gz.sig",
            ),
        ),
    ),
    (
        (
            "MACOS_ARM64",
            "* [macOS Apple Silicon]({0}) (ARM64) ([signature]({1}))",
            (
                "LLVM-{release}-macOS-ARM64.tar.xz",
                "LLVM-{release}-macOS-ARM64.tar.xz.jsonl",
            ),
        ),
        (
            "MACOS_X86",
            "* [macOS Intel]({0}) (x86-64) ([signature]({1}))",
            (
                "LLVM-{release}-macOS-X64.tar.xz",
                "LLVM-{release}-macOS-X64.tar.xz.jsonl",
            ),
        ),
    ),
    (
        (
            "WINDOWS_X64",
            "* Windows x64 (64-bit): [installer]({0}) ([signature]({1})), [archive]({2}) ([signature]({3}))",
            (
                "LLVM-{release}-win64.exe",
                "LLVM-{release}-win64.exe.sig",
                "clang+llvm-{release}-x86_64-pc-windows-msvc.tar.xz",
                "clang+llvm-{release}-x86_64-pc-windows-msvc.tar.xz.sig",
            ),
        ),
        (
            "WINDOWS_X86",
            "* Windows x86 (32-bit): [installer]({0}) ([signature]({1}))",
            ("LLVM-{release}-win32.exe", "LLVM-{release}-win32.exe.sig"),
        ),
        (
            "WINDOWS_ARM64",
            "* Windows on Arm (ARM64): [installer]({0}) ([signature]({1})), [archive]({2}) ([signature]({3}))",
            (
                "LLVM-{release}-woa64.exe",
                "LLVM-{release}-woa64.exe.sig",
                "clang+llvm-{release}-aarch64-pc-windows-msvc.tar.xz",
                "clang+llvm-{release}-aarch64-pc-windows-msvc.tar.xz.sig",
            ),
        ),
    ),
)


def generate_download_links(release):
    base_url = (
        f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{release}/"
    )
    markdown_lines = []

    for section in release_links:
        for line in section:
            comment_tag, format_string, files = line
            markdown_line = f"<!-- {comment_tag} "
            files = [base_url + f.format(release=release) for f in files]
            markdown_line += format_string.format(*files)
            markdown_line += " -->"
            markdown_lines.append(markdown_line)

        # Blank line between each section.
        markdown_lines.append("")

    return "\n".join(markdown_lines)


def create_release(repo, release, tag=None, name=None, message=None):
    if not tag:
        tag = "llvmorg-{}".format(release)

    if not name:
        name = "LLVM {}".format(release)

    if not message:
        # Note that these lines are not length limited because if we do so, GitHub
        # assumes that should be how it is laid out on the page. We want GitHub to
        # do the reflowing for us instead.
        download_links = generate_download_links(release)
        message = dedent(
            f"""\
## LLVM {release} Release

{download_links}
Download links for common platforms will appear above once builds have completed, if they are available. Check the full list of release packages at the bottom of this release page if you do not find a link above.

If you do not find a release package for your platform, you may be able to find a community built package on the LLVM Discourse forum thread for this release. Remember that these are built by volunteers and may not always be available. If you rely on a platform or configuration that is not one of the defaults, we suggest you use the binaries that your platform provides, or build your own release packages.

## Package Types

Each platform has one binary release package. The file name starts with either `LLVM-` or `clang+llvm-` and ends with the platform's name. For example, `LLVM-{release}-Linux-ARM64.tar.xz` contains LLVM binaries for Arm64 Linux.

Except for Windows. Where `LLVM-*.exe` is an installer intended for using LLVM as a toolchain and the archive `clang+llvm-` contains the contents of the installer, plus libraries and tools not normally used in a toolchain. You most likely want the `LLVM-` installer, unless you are developing software which itself uses LLVM, in which case choose `clang+llvm-`.

In addition, source archives are available:
* To get all the `llvm-project` source code for this release, choose `llvm-project-{release}.src.tar.xz`.
* `test-suite-{release}.src.tar.xz` is an archive of the [LLVM Test Suite](https://github.com/llvm/llvm-test-suite)) for this release.

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
        )

    prerelease = True if "rc" in release else False

    repo.create_git_release(tag=tag, name=name, message=message, prerelease=prerelease)


def upload_files(repo, release, files):
    release = repo.get_release("llvmorg-{}".format(release))
    for f in files:
        print("Uploading {}".format(f))
        release.upload_asset(f)
        print("Done")


def uncomment_download_links(repo, release_version):
    release = repo.get_release(f"llvmorg-{release_version}")

    # At this point any automatic builds have finished and if
    # they succeeded, uploaded files to the release assets.
    release_assets = set([a.name for a in release.assets])
    print("Found release assets: ", release_assets)

    new_message = []
    modified = False
    for line in release.body.splitlines():
        # All hidden download links are of the form:
        # <!-- <some unique tag> <markdown content> -->
        if not line.startswith("<!--"):
            new_message.append(line)
            continue

        for section in release_links:
            for comment_tag, _, files in section:
                if not comment_tag in line:
                    continue

                print(f'Found link line "{comment_tag}":')
                files = set([f.format(release=release_version) for f in files])
                print("  Files required:", files)
                if files.issubset(release_assets):
                    print("  All files present, revealing link line.")
                    line = (
                        line.replace("<!--", "")
                        .replace(comment_tag, "")
                        .replace("-->", "")
                        .strip()
                    )
                    modified = True
                else:
                    print(
                        "  These files are not present:",
                        files.difference(release_assets),
                    )
                    print("  Link line will remain hidden.")

        new_message.append(line)

    if modified:
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
