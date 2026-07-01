@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s

  .text
  .thumb
@ CHECK: error: too few operands for instruction
  strd
@ CHECK: error: too few operands for instruction
  ldrd
@ CHECK: error: too few operands for instruction
  strd r0
@ CHECK: error: too few operands for instruction
  ldrd r0
@ CHECK: error: operand must be a register in range [r0, r15]
  strd s0, [r0]
@ CHECK: error: operand must be a register in range [r0, r15]
  ldrd s0, [r0]
  .arm
@ CHECK: error: too few operands for instruction
  strd
@ CHECK: error: too few operands for instruction
  ldrd
@ CHECK: error: too few operands for instruction
  strd r0
@ CHECK: error: too few operands for instruction
  ldrd r0
@ CHECK: error: operand must be a register in range [r0, r15]
  strd s0, [r0]
@ CHECK: error: operand must be a register in range [r0, r15]
  ldrd s0, [r0]
