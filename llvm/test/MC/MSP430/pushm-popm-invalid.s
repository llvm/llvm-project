// Test for errors when PUSHM/POPM instructions are misused.

// RUN: not llvm-mc -triple msp430 -mcpu=msp430x < %s 2>&1 | FileCheck %s
pushm:
  pushm #0, r4
  pushm #17, r4
  pushm #17, r15
  pushm r4
  pushm #1, 0(r4)
  pushm #1, &foo
// CHECK: :9: error: invalid operand for instruction
// CHECK: :9: error: invalid operand for instruction
// CHECK: :9: error: invalid operand for instruction
// CHECK: :9: error: invalid operand for instruction
// CHECK: :13: error: invalid operand for instruction
// CHECK: :13: error: invalid operand for instruction


popm:
  popm #0, r4
  popm #17, r4
  popm #17, r15
  popm r4
  popm #1, 0(r4)
  popm #1, &foo
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :12: error: invalid operand for instruction
// CHECK: :12: error: invalid operand for instruction

// The register range must not extend below r0.
  pushm #5, r3
// CHECK: :[[@LINE-1]]:13: error: register range extends below r0
  popm #16, r14
// CHECK: :[[@LINE-1]]:13: error: register range extends below r0
