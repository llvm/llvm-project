#!/bin/bash

# This script will sync github.com/llvm/llvm-project with
# github.com/llvm/llvm-project-release-prs and try to merge
# the changes in the release branch.

set -e
set -x

# We should always get the branch from the environment.
# But otherwise just default to something. We can probably
# have a better default here?
RELEASE_BRANCH="${RELEASE_BRANCH:-release/16.x}"

# We will add two remotes here:
#  main - which will point to the main llvm-project repo
#  release - which will point to the release-prs repo
# The remotes will use random strings to avoid
# collisions
MAIN_REMOTE=$(uuidgen)
RELEASE_REMOTE=$(uuidgen)
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

git remote add $MAIN_REMOTE "https://github.com/llvm/llvm-project"
git remote add $RELEASE_REMOTE "https://github.com/llvm/llvm-project-release-prs"

# Make sure we are up to date on all our repos first
git fetch $MAIN_REMOTE
git fetch $RELEASE_REMOTE

# Create our sync branch. Starting with the main
# repo first since it's important to get those
# changes
MERGE_BRANCH=$(uuidgen)
git switch -c $MERGE_BRANCH $MAIN_REMOTE/$RELEASE_BRANCH

# Merge changes from the release repo
git merge --ff-only $RELEASE_REMOTE/$RELEASE_BRANCH

if ! git diff-index --quiet $MAIN_REMOTE/$RELEASE_BRANCH; then
  echo "Changes in the release remote - pushing to main remote"
  git push $MAIN_REMOTE $MERGE_BRANCH:$RELEASE_BRANCH
fi

# Before we merge back into the release repo
# let's update to make sure nothing has been
# pushed to either repo while we do this work.
# Most of the time this won't do anything, and
# the real solution would instead be to fetch
# in a loop if pushing fails. But that's a very
# tiny edge-case, so let's not complicate this.
git fetch $MAIN_REMOTE
git fetch $RELEASE_REMOTE

# And merge all the new data to the current branch
git merge --ff-only $MAIN_REMOTE/$RELEASE_BRANCH

# If anything changed let's merge it
if ! git diff-index --quiet $RELEASE_REMOTE/$RELEASE_BRANCH; then
  echo "Changes in main - pushing to release"
  git push $RELEASE_REMOTE $MERGE_BRANCH:$RELEASE_BRANCH
fi

# Cleanup - enable for debug
if false; then
  git remote remove $RELEASE_REMOTE
  git remote remove $MAIN_REMOTE

  git switch $CURRENT_BRANCH
  git branch -D $MERGE_BRANCH
fi
