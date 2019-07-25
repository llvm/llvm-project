#!/usr/bin/env python3
# ===-- github-upload-release.py  ------------------------------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
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
#===------------------------------------------------------------------------===#


import argparse
import github

def create_release(repo, release, tag = None, name = None, message = None):
    if not tag:
        tag = 'llvmorg-{}'.format(release)

    if not name:
        name = 'LLVM {}'.format(release)

    if not message:
        message = 'LLVM {} Release'.format(release)

    prerelease = True if "rc" in release else False

    repo.create_git_release(tag = tag, name = name, message = message,
                            prerelease = prerelease)

def upload_files(repo, release, files):
    release = repo.get_release('llvmorg-{}'.format(release))
    for f in files:
        print('Uploading {}'.format(f))
        release.upload_asset(f)
        print("Done")
    


parser = argparse.ArgumentParser()
parser.add_argument('command', type=str, choices=['create', 'upload'])

# All args
parser.add_argument('--token', type=str)
parser.add_argument('--release', type=str)

# Upload args
parser.add_argument('--files', nargs='+', type=str)


args = parser.parse_args()

github = github.Github(args.token)
llvm_repo = github.get_organization('llvm').get_repo('llvm-project')

if args.command == 'create':
    create_release(llvm_repo, args.release)
if args.command == 'upload':
    upload_files(llvm_repo, args.release, args.files)
