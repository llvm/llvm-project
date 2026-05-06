#!/bin/bash
#
# query-test.sh
# `2>&1` redirects stderr (where errors and diagnostics are printed) to stdout
# so it can be piped to grep.
#
# Note mlir-opt needs to be on your path, or else replace mlir-opt with the
# absolute path to the binary. If mlir-opt cannot be found, this interestingness
# test will report "uninteresting on the first run, and mlir-reduce will
# report that the input is not marked as interesting.
#
mlir-opt $1 2>&1 | grep "arith.select"

# grep's exit code is 0 if the queried string is found
if [[ $? -eq 0 ]]; then
  exit 1
else
  exit 0
fi
