#!/bin/sh
# This script is used by the mlir-reduce documentation example test.
mlir-opt $1 2>&1 | grep "arith.select" > /dev/null
if [ $? -eq 0 ]; then
  exit 1
else
  exit 0
fi
