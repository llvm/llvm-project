# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=x86-64 < %s 2>&1 | FileCheck %s

# This test checks that https://github.com/llvm/llvm-project/issues/62528 is resolved.
foo:
  pushq   %rbp

# CHECK-NOT:      <stdin>:4:1: error: symbol 'foo' is already defined

