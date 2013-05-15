#!/bin/bash

if ! which clang-format; then
    echo "Error: cannot find clang-format in your path"
    exit 1
fi

OK=0

for ARG in "$@"
  do
    clang-format $ARG | diff -u $ARG -

    if [[ $? -eq 1 ]]; then
      OK=1
    fi
  done

if [[ $OK -eq "1" ]]; then
  echo "Error: clang-format reported formatting differences"
  exit 1
else
  echo "OK: clang-format reported no formatting differences"
  exit 0
fi

