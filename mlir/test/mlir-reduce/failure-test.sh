#!/bin/sh
# Tests for the keyword "failure" in the stderr of the optimization pass
mlir-opt $1 -test-mlir-reducer > /tmp/stdout.$$ 2>/tmp/stderr.$$

if [ $? -ne 0 ] && grep 'failure' /tmp/stderr.$$; then
  exit 1
  #Interesting behavior
else 
  exit 0
fi
