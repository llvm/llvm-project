# RUN: not llvm-mca %s -mtriple=x86_64-unknown-unknown -mcpu=foo -o /dev/null 2>&1 | FileCheck %s

# CHECK: 'foo' is not a recognized processor for this target (ignoring processor)
