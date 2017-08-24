#!/usr/bin/env bash
#===- llvm/utils/docker/scripts/build_install_llvm.sh ---------------------===//
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
Usage: build_install_llvm.sh [options] -- [cmake-args]

Checkout svn sources and run cmake with the specified arguments. Used
inside docker container.
Passes additional -DCMAKE_INSTALL_PREFIX and archives the contents of
the directory to /tmp/clang.tar.gz.

Available options:
  -h|--help           show this help message
  -b|--branch         svn branch to checkout, i.e. 'trunk',
                      'branches/release_40'
                      (default: 'trunk')
  -r|--revision       svn revision to checkout
  -p|--llvm-project   name of an svn project to checkout. Will also add the
                      project to a list LLVM_ENABLE_PROJECTS, passed to CMake.
                      For clang, please use 'clang', not 'cfe'.
                      Project 'llvm' is always included and ignored, if
                      specified.
                      Can be specified multiple times.
  -i|--install-target name of a cmake install target to build and include in
                      the resulting archive. Can be specified multiple times.
Required options: At least one --install-target.

All options after '--' are passed to CMake invocation.
EOF
}

LLVM_SVN_REV=""
LLVM_BRANCH=""
CMAKE_ARGS=""
CMAKE_INSTALL_TARGETS=""
# We always checkout llvm
LLVM_PROJECTS="llvm"
CMAKE_LLVM_ENABLE_PROJECTS=""
CLANG_TOOLS_EXTRA_ENABLED=0

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

function append_project() {
  local PROJ="$1"

  LLVM_PROJECTS="$LLVM_PROJECTS $PROJ"
  if [ "$CMAKE_LLVM_ENABLE_PROJECTS" != "" ]; then
    CMAKE_LLVM_ENABLE_PROJECTS="$CMAKE_LLVM_ENABLE_PROJECTS;$PROJ"
  else
    CMAKE_LLVM_ENABLE_PROJECTS="$PROJ"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--revision)
      shift
      LLVM_SVN_REV="$1"
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

      if [ "$PROJ" == "clang-tools-extra" ]; then
        if [ $CLANG_TOOLS_EXTRA_ENABLED -ne 0 ]; then
          echo "Project 'clang-tools-extra' is already enabled, ignoring extra occurences."
        else
          CLANG_TOOLS_EXTRA_ENABLED=1
        fi

        continue
      fi

      if ! contains_project "$PROJ" ; then
        append_project "$PROJ"
      else
        echo "Project '$PROJ' is already enabled, ignoring extra occurences."
      fi
      ;;
    -i|--install-target)
      shift
      CMAKE_INSTALL_TARGETS="$CMAKE_INSTALL_TARGETS $1"
      shift
      ;;
    --)
      shift
      CMAKE_ARGS="$*"
      shift $#
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

if [ "$CMAKE_INSTALL_TARGETS" == "" ]; then
  echo "No install targets. Please pass one or more --install-target."
  exit 1
fi

if [ $CLANG_TOOLS_EXTRA_ENABLED -ne 0 ]; then
  if ! contains_project "clang"; then
    echo "Project 'clang-tools-extra' was enabled without 'clang'."
    echo "Adding 'clang' to a list of projects."

    append_project "clang"
  fi
fi

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

CLANG_BUILD_DIR=/tmp/clang-build
CLANG_INSTALL_DIR=/tmp/clang-install

mkdir "$CLANG_BUILD_DIR"

# Get the sources from svn.
echo "Checking out sources from svn"
mkdir "$CLANG_BUILD_DIR/src"
for LLVM_PROJECT in $LLVM_PROJECTS; do
  if [ "$LLVM_PROJECT" == "clang" ]; then
    SVN_PROJECT="cfe"
  else
    SVN_PROJECT="$LLVM_PROJECT"
  fi

  echo "Checking out https://llvm.org/svn/llvm-project/$SVN_PROJECT to $CLANG_BUILD_DIR/src/$LLVM_PROJECT"
  svn co -q $SVN_REV_ARG \
    "https://llvm.org/svn/llvm-project/$SVN_PROJECT/$LLVM_BRANCH" \
    "$CLANG_BUILD_DIR/src/$LLVM_PROJECT"
done

if [ $CLANG_TOOLS_EXTRA_ENABLED -ne 0 ]; then
  echo "Checking out https://llvm.org/svn/llvm-project/clang-tools-extra to $CLANG_BUILD_DIR/src/clang/tools/extra"
  svn co -q $SVN_REV_ARG \
    "https://llvm.org/svn/llvm-project/clang-tools-extra/$LLVM_BRANCH" \
    "$CLANG_BUILD_DIR/src/clang/tools/extra"
fi

mkdir "$CLANG_BUILD_DIR/build"
pushd "$CLANG_BUILD_DIR/build"

# Run the build as specified in the build arguments.
echo "Running build"
cmake -GNinja \
  -DCMAKE_INSTALL_PREFIX="$CLANG_INSTALL_DIR" \
  -DLLVM_ENABLE_PROJECTS="$CMAKE_LLVM_ENABLE_PROJECTS" \
  $CMAKE_ARGS \
  "$CLANG_BUILD_DIR/src/llvm"
ninja $CMAKE_INSTALL_TARGETS

popd

# Pack the installed clang into an archive.
echo "Archiving clang installation to /tmp/clang.tar.gz"
cd "$CLANG_INSTALL_DIR"
tar -czf /tmp/clang.tar.gz *

# Cleanup.
rm -rf "$CLANG_BUILD_DIR" "$CLANG_INSTALL_DIR"

echo "Done"
