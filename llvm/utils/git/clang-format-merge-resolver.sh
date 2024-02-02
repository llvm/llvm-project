#!/usr/bin/env bash

# One-time setup:
#     git config mergetool.llvm-reformat.cmd 'llvm/utils/git/clang-format-merge-resolver.sh "$BASE" "$LOCAL" "$REMOTE" "$MERGED" '
#
# Usage:
#     If clang-format is not on PATH:
#         export CLANG_FORMAT_PATH=/path/to/clang-format
#     After you have merged a reformatting commit and it has conflicts, run:
#         git mergetool --tool=llvm-reformat

# This script is based on libcxx/utils/clang-format-merge-driver.sh
# which sadly cannot simply be used.

# Find clang-format the same way code-format-helper.py does.
if [ -x "$CLANG_FORMAT_PATH" ]; then
    CLANG_FORMAT=$CLANG_FORMAT_PATH
else
    CLANG_FORMAT=`which clang-format`
    if [ ! $? ]; then
        exit -1
    fi
fi

# Path to the file's contents at the ancestor's version.
base="$1"

# Path to the file's contents at the current version.
current="$2"

# Path to the file's contents at the other branch's version (for nonlinear histories, there might be multiple other branches).
other="$3"

# The path of the file in the repository.
path="$4"

$CLANG_FORMAT --style=file --assume-filename="$path" < "$base" > "$base.tmp"
mv "$base.tmp" "$base"

$CLANG_FORMAT --style=file --assume-filename="$path" < "$current" > "$current.tmp"
mv "$current.tmp" "$current"

$CLANG_FORMAT --style=file --assume-filename="$path" < "$other" > "$other.tmp"
mv "$other.tmp" "$other"

git merge-file -Lcurrent -Lbase -Lother "$current" "$base" "$other"
STATUS=$?
if [ $STATUS ]; then
    mv "$current" "$path"
fi
exit $STATUS
