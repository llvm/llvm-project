#!/bin/bash

set -euo pipefail

# "script-and-regex.regex": "/^(?P<severity>.*?)\n(?P<message>.*?)\n(?P<line>\\d),(?P<char>\\d)(\n(?P<original>.*?)>>>>\n(?P<replacement>.*?)<<<<?)$/s",

# Arcanist linter that invokes clang-format.
# stdout from this script is parsed into a regex and used by Arcanist.
# https://secure.phabricator.com/book/phabricator/article/arcanist_lint_script_and_regex/

# To skip running all linters when creating/updating a diff, use `arc diff --nolint`.

if ! hash git-clang-format >/dev/null; then
  # advice severity level is completely non-disruptive.
  # switch to warning or error if you want to prompt the user.
  echo "advice"
  echo "git-clang-format not found in user's PATH; not linting file."
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
# message on line 1, char 1 with file content edited using git-clang-format.
if git rev-parse --git-dir >/dev/null; then
  arc_base_commit=$(arc which --show-base)
  # An alternative is to use git-clang-format.
  >&2 git-clang-format --quiet --force --style file "${arc_base_commit}"
else
  >&2 echo "repo is expected to be a git directory"
fi

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
