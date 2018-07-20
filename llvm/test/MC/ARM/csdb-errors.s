// RUN: not llvm-mc -triple   armv8a-none-eabi %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple thumbv8a-none-eabi %s 2>&1 | FileCheck %s

  it eq
  csdbeq
// CHECK: error: instruction 'csdb' is not predicable, but condition code specified
