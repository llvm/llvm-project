#!/usr/bin/env bash

# Usage:
#     If clang-format is not on PATH:
#         export CLANG_FORMAT_PATH=/path/to/clang-format
#     Run this script to resolve the formatting conflicts and leave only
#     the non-formatting conflicts for you to resolve manually.

# Find the .git directory.
GIT_DIR=$(git rev-parse --git-dir)

# Are we in the midst of a merge? If not, this is the wrong tool.
if [ ! -e $GIT_DIR/MERGE_HEAD ]; then
    echo Not doing a merge?
    exit -1
fi

# Find the "current" "other" and "base" commits.
# The commit we are merging into.
read CURRENT < $GIT_DIR/ORIG_HEAD
# The commit being merged into CURRENT.
read OTHER < $GIT_DIR/MERGE_HEAD
# Where it all started.
BASE=$(git merge-base $CURRENT $OTHER)

# Set up a place to keep temp files.
MYTEMP=$(mktemp -d)
trap 'rm -rf $MYTEMP' EXIT

# Find clang-format the same way code-format-helper.py does.
if [ -x "$CLANG_FORMAT_PATH" ]
then
    CLANG_FORMAT=$CLANG_FORMAT_PATH
else
    CLANG_FORMAT=$(which clang-format)
    if [ ! $? ]
    then
        echo clang-format not found on PATH
        exit -1
    fi
fi

# resolve_one_file will perform formatting-conflict resolution on one file.
# If any conflicts remain, informs the user, otherwise will git-add the
# resolved file.
resolve_one_file() {
    file=$1
    echo Resolving "$file"...

    # Get formatted copies of the base, current, and other files.
    git show "$BASE:$file"    | $CLANG_FORMAT --style=file --assume-filename="$file" > "$MYTEMP/base"
    git show "$OTHER:$file"   | $CLANG_FORMAT --style=file --assume-filename="$file" > "$MYTEMP/other"
    git show "$CURRENT:$file" | $CLANG_FORMAT --style=file --assume-filename="$file" > "$MYTEMP/current"

    # Merge the formatted files and report failures.
    git merge-file -Lcurrent -Lbase -Lother "$MYTEMP/current" "$MYTEMP/base" "$MYTEMP/other"
    STATUS=$?
    if [ $STATUS -lt 0 ]
    then
        echo git merge-file failed for "$file"
    else
        mv "$MYTEMP/current" "$file"
        if [ $STATUS -eq 0 ]
        then
            git add "$file"
        else
            echo Conflicts remain in $file
        fi
    fi
}

# Find all the conflicted files, and operate on each one.
# `git status --porcelain` identifies conflicted files with 'UU ' prefix.
# No other prefix is relevant here.
git status --porcelain | grep '^UU ' | cut -c 4- | \
    while read -r file
    do
        resolve_one_file "$file"
    done
