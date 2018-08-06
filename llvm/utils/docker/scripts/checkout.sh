#!/usr/bin/env bash
#===- llvm/utils/docker/scripts/checkout.sh ---------------------===//
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===-----------------------------------------------------------------------===//

set -e

function show_usage() {
  cat << EOF
Usage: checkout.sh [options]

Checkout svn sources into /tmp/clang-build/src. Used inside a docker container.

Available options:
  -h|--help           show this help message
  -b|--branch         svn branch to checkout, i.e. 'trunk',
                      'branches/release_40'
                      (default: 'trunk')
  -r|--revision       svn revision to checkout
  -c|--cherrypick     revision to cherry-pick. Can be specified multiple times.
                      Cherry-picks are performed in the sorted order using the
                      following command:
                      'svn patch <(svn diff -c \$rev)'.
  -p|--llvm-project   name of an svn project to checkout.
                      For clang, please use 'clang', not 'cfe'.
                      Project 'llvm' is always included and ignored, if
                      specified.
                      Can be specified multiple times.
EOF
}

LLVM_SVN_REV=""
CHERRYPICKS=""
LLVM_BRANCH=""
# We always checkout llvm
LLVM_PROJECTS="llvm"

function contains_project() {
  local TARGET_PROJ="$1"
  local PROJ
  for PROJ in $LLVM_PROJECTS; do
    if [ "$PROJ" == "$TARGET_PROJ" ]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--revision)
      shift
      LLVM_SVN_REV="$1"
      shift
      ;;
    -c|--cherrypick)
      shift
      CHERRYPICKS="$CHERRYPICKS $1"
      shift
      ;;
    -b|--branch)
      shift
      LLVM_BRANCH="$1"
      shift
      ;;
    -p|--llvm-project)
      shift
      PROJ="$1"
      shift

      if [ "$PROJ" == "cfe" ]; then
        PROJ="clang"
      fi

      if ! contains_project "$PROJ" ; then
        if [ "$PROJ" == "clang-tools-extra" ] && [ ! contains_project "clang" ]; then
          echo "Project 'clang-tools-extra' specified before 'clang'. Adding 'clang' to a list of projects first."
          LLVM_PROJECTS="$LLVM_PROJECTS clang"
        fi
        LLVM_PROJECTS="$LLVM_PROJECTS $PROJ"
      else
        echo "Project '$PROJ' is already enabled, ignoring extra occurrences."
      fi
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
  esac
done

if [ "$LLVM_BRANCH" == "" ]; then
  LLVM_BRANCH="trunk"
fi

if [ "$LLVM_SVN_REV" != "" ]; then
  SVN_REV_ARG="-r$LLVM_SVN_REV"
  echo "Checking out svn revision r$LLVM_SVN_REV."
else
  SVN_REV_ARG=""
  echo "Checking out latest svn revision."
fi

# Sort cherrypicks and remove duplicates.
CHERRYPICKS="$(echo "$CHERRYPICKS" | xargs -n1 | sort | uniq | xargs)"

function apply_cherrypicks() {
  local CHECKOUT_DIR="$1"

  [ "$CHERRYPICKS" == "" ] || echo "Applying cherrypicks"
  pushd "$CHECKOUT_DIR"

  # This function is always called on a sorted list of cherrypicks.
  for CHERRY_REV in $CHERRYPICKS; do
    echo "Cherry-picking r$CHERRY_REV into $CHECKOUT_DIR"

    local PATCH_FILE="$(mktemp)"
    svn diff -c $CHERRY_REV > "$PATCH_FILE"
    svn patch "$PATCH_FILE"
    rm "$PATCH_FILE"
  done

  popd
}

CLANG_BUILD_DIR=/tmp/clang-build

# Get the sources from svn.
echo "Checking out sources from svn"
mkdir -p "$CLANG_BUILD_DIR/src"
for LLVM_PROJECT in $LLVM_PROJECTS; do
  if [ "$LLVM_PROJECT" == "clang" ]; then
    SVN_PROJECT="cfe"
  else
    SVN_PROJECT="$LLVM_PROJECT"
  fi

  if [ "$SVN_PROJECT" != "clang-tools-extra" ]; then
    CHECKOUT_DIR="$CLANG_BUILD_DIR/src/$LLVM_PROJECT"
  else
    CHECKOUT_DIR="$CLANG_BUILD_DIR/src/clang/tools/extra"
  fi

  echo "Checking out https://llvm.org/svn/llvm-project/$SVN_PROJECT to $CHECKOUT_DIR"
  svn co -q $SVN_REV_ARG \
    "https://llvm.org/svn/llvm-project/$SVN_PROJECT/$LLVM_BRANCH" \
    "$CHECKOUT_DIR"

  # We apply cherrypicks to all repositories regardless of whether the revision
  # changes this repository or not. For repositories not affected by the
  # cherrypick, applying the cherrypick is a no-op.
  apply_cherrypicks "$CHECKOUT_DIR"
done

CHECKSUMS_FILE="/tmp/checksums/checksums.txt"

if [ -f "$CHECKSUMS_FILE" ]; then
  echo "Validating checksums for LLVM checkout..."
  python "$(dirname $0)/llvm_checksum/llvm_checksum.py" -c "$CHECKSUMS_FILE" \
    --partial --multi_dir "$CLANG_BUILD_DIR/src"
else
  echo "Skipping checksumming checks..."
fi

echo "Done"
