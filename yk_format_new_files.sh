#!/bin/sh
#
# Format any new files that we have added in our fork.

set -e

# The LLVM version tag we are forked from.
FORKED_LLVM_VERSION=`git describe --abbrev=0 --match 'llvmorg-*'`

git diff --name-only --diff-filter=A ${FORKED_LLVM_VERSION} \
    `git branch --show-current` | egrep '\.(cpp|h)$' | xargs clang-format -i
