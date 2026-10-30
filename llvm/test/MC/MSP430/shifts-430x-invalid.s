// Tests to check for incorrect use of MSP430X shift instructions.
// RUN: not llvm-mc --triple=msp430 -mcpu=msp430x < %s 2>&1 | FileCheck %s

foo:
// Check for an error when the repetition count is too large.
  rrum #5, r12
  rram #5, r12
  rlam #5, r12
  rrcm #5, r12
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction

// Check for an error when RxxX shifts have a shift amount operand.
  rrcx #3, r12
  rrux #3, r12
  rrax #3, r12
  rlax #3, r12
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
// CHECK: :8: error: invalid operand for instruction
