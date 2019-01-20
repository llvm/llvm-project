#!/bin/bash
#===-- merge-git.sh - Merge commit to the stable branch --------------------===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# This script will merge an svn revision to a git repo using git-svn while
# preserving the svn commit message.
# 
# NOTE: This script has only been tested with the per-project git repositories
# and not with the monorepo.
#
# In order to use this script, you must:
# 1) Checkout the stable branch you would like to merge the revision into.
# 2) Correctly configure the branch as an svn-remote by adding the following to
# your .git/config file for your git repo (replace xy with the major/minor
# version of the release branch. e.g. release_50 or release_60):
#
#[svn-remote "release_xy"]
#url = https://llvm.org/svn/llvm-project/llvm/branches/release_xy
#fetch = :refs/remotes/origin/release_xy
#
# Once the script completes successfully, you can push your changes with
# git-svn dcommit
#
#===------------------------------------------------------------------------===#


usage() {
    echo "usage: `basename $0` [OPTIONS]"
    echo "  -rev NUM       The revision to merge into the project"
}

while [ $# -gt 0 ]; do
    case $1 in
        -rev | --rev | -r )
            shift
            rev=$1
            ;;
        -h | -help | --help )
            usage
            ;;
        * )
            echo "unknown option: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
    shift
done

if [ -z "$rev" ]; then
    echo "error: need to specify a revision"
    echo
    usage
    exit 1
fi

# Rebuild revision map
git svn find-rev r$rev origin/master &>/dev/null

git_hash=`git svn find-rev r$rev origin/master`

if [ -z "$git_hash" ]; then
    echo "error: could not determine git commit for r$rev"
    exit 1
fi

commit_msg=`svn log -r $rev https://llvm.org/svn/llvm-project/`
ammend="--amend"

git cherry-pick $git_hash
if [ $? -ne 0 ]; then
  echo ""
  echo "** cherry-pick failed enter 'e' to exit or 'c' when you have finished resolving the conflicts:"
  read option
  case $option in
    c)
      ammend=""
      ;;
    *)
      exit 1
      ;;
  esac
fi
         
git commit $ammend -m "Merging r$rev:" -m "$commit_msg"
