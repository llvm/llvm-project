#!/bin/sh
#
# Format any new files that we have added in our fork.

set -e

clang/tools/clang-format/git-clang-format origin/main
