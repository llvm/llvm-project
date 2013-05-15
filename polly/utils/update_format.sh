#!/bin/bash

if ! which clang-format; then
    echo "Error: cannot find clang-format in your path"
    exit 1
fi

for ARG in "$@"
  do
    clang-format -i $ARG
  done
