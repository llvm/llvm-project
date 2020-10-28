#!/bin/bash

set -euo pipefail

# "script-and-regex.regex": "/^(?P<severity>.*?)\n(?P<message>.*?)\n(?P<line>\\d),(?P<char>\\d)(\n(?P<original>.*?)>>>>\n(?P<replacement>.*?)<<<<?)$/s",

# Arcanist linter that invokes clang-format via clang/tools/clang-format/clang-format-diff.py
# stdout from this script is parsed into a regex and used by Arcanist.
# https://secure.phabricator.com/book/phabricator/article/arcanist_lint_script_and_regex/

# To skip running all linters when creating/updating a diff, use `arc diff --nolint`.

# advice severity level is completely non-disruptive.
# switch to warning or error if you want to prompt the user.
if ! hash clang-format >/dev/null; then
  echo "advice"
  echo "clang-format not found in user's PATH; not linting file."
  echo "===="
  exit 0
fi
if ! git rev-parse --git-dir >/dev/null; then
  echo "advice"
  echo "not in git repostitory; not linting file."
  echo "===="
  exit 0
fi

src_file="${1}"
original_file="$(mktemp)"
formatted_file="$(mktemp)"
readonly src_file
readonly original_file
readonly formatted_file
cp -p "${src_file}" "${original_file}"
cp -p "${src_file}" "${formatted_file}"

cleanup() {
  rc=$?
  rm "${formatted_file}" "${original_file}"
  exit ${rc}
}
trap 'cleanup' INT HUP QUIT TERM EXIT

# Arcanist can filter out lint messages for unchanged lines, but for that, we
# need to generate line by line lint messages. Instead, we generate one lint
# message on line 1, char 1 with file content edited using clang-format-diff.py
#
# We do not use git-clang-format because it wants to modify the index,
# and arc is already holding the lock.
#
# We do not look for clang-format-diff or clang-format-diff.py in the PATH
# because whether/how these are installed differs between distributions,
# and we have an executable copy in the tree anyway.
arc_base_commit=$(arc which --show-base)
git diff-index -U0 "${arc_base_commit}" "${src_file}" \
  | clang/tools/clang-format/clang-format-diff.py -style file -i -p1

cp -p "${src_file}" "${formatted_file}"
cp -p "${original_file}" "${src_file}"
if ! diff -q "${src_file}" "${formatted_file}" > /dev/null ; then
  echo "autofix"
  echo "clang-format suggested style edits found:"
  echo "1,1"  # line,char of start of replacement.
  cat "${src_file}"
  echo ">>>>"
  cat "${formatted_file}"
  echo "<<<<"
fi
