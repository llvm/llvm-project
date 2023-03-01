#!/bin/sh

# Create temporary files that are automatically deleted after the script's
# execution.
stdout_file=$(mktemp /tmp/stdout.XXXXXX)
stderr_file=$(mktemp /tmp/stderr.XXXXXX)

# Tests for the keyword "failure" in the stderr of the optimization pass
mlir-opt $1 -test-mlir-reducer > $stdout_file 2> $stderr_file

if [ $? -ne 0 ] && grep 'failure' $stderr_file; then
  exit 1
  #Interesting behavior
else 
  exit 0
fi
