#!/bin/sh

file=$1

if grep -q "arith.constant 1 : i32" $file && grep -q "arith.constant 2 : i32" $file && grep -q "arith.constant 3 : i32" $file; then
  exit 1
fi
