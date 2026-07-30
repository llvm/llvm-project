// RUN: not llvm-mc -triple   armv8a-none-eabi %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple thumbv8a-none-eabi %s 2>&1 | FileCheck %s -check-prefix=THUMB

  it eq
  csdbeq

  it eq
  ssbbeq

  it eq
  pssbbeq

  it eq
  hinteq #20

  it eq
  dsbeq #0

  it eq
  dsbeq #4

// CHECK: error: instruction 'csdb' is not predicable, but condition code specified
// CHECK: error: instruction 'ssbb' is not predicable, but condition code specified
// CHECK: error: instruction 'pssbb' is not predicable, but condition code specified
// CHECK: error: instruction 'csdb' is not predicable, but condition code specified
// CHECK: error: instruction 'dsb' is not predicable, but condition code specified
// CHECK: error: instruction 'dsb' is not predicable, but condition code specified

// THUMB: error: instruction 'csdb' is not predicable, but condition code specified
// THUMB: error: instruction 'ssbb' is not predicable, but condition code specified
// THUMB: error: instruction 'pssbb' is not predicable, but condition code specified
// THUMB: error: instruction 'csdb' is not predicable, but condition code specified
// THUMB: error: instruction 'ssbb' is not predicable, but condition code specified
// THUMB: error: instruction 'pssbb' is not predicable, but condition code specified
