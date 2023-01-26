#!/bin/sh
#
# Format yk-specific folders.
#
# In the past we tried using git-clang-format against upstream LLVM. This
# worked well until you do an upstream sync. Since upstream's changes are not
# consistently formatted, it meant that CI would always fail.
#
# For now we just format folders that contain only new files that we've
# introduced for Yk.
#
# This script must be run in the root of the repository.

set -e

YK_DIRS="./clang/test/Yk ./llvm/lib/Transforms/Yk ./llvm/include/llvm/Transforms/Yk"

for dir in ${YK_DIRS}; do
    find ${dir} -type f -iname '*.cpp' -or -iname '*.h' -or -iname '*.c' | \
        xargs ./build/bin/clang-format -i
done
