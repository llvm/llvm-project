#!/bin/sh

# Create temporary files and delete them when script exits.
stdout_file=$(mktemp /tmp/stdout.XXXXXX)
stderr_file=$(mktemp /tmp/stderr.XXXXXX)
trap 'rm -f "$stdout_file" "$stderr_file"' EXIT

# Tests for the keyword "failure" in the stderr of the optimization pass
mlir-opt $1 -test-mlir-reducer > $stdout_file 2> $stderr_file

if [ $? -ne 0 ] && grep 'failure' $stderr_file; then
  exit 1
  #Interesting behavior
else 
  exit 0
fi
