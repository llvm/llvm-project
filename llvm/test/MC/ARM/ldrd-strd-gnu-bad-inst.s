@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s

  .text
  .thumb
@ CHECK: error: invalid instruction
  strd
@ CHECK: error: invalid instruction
  ldrd
@ CHECK: error: invalid instruction
  strd r0
@ CHECK: error: invalid instruction
  ldrd r0
@ CHECK: error: invalid instruction
  strd s0, [r0]
@ CHECK: error: invalid instruction
  ldrd s0, [r0]
  .arm
@ CHECK: error: invalid instruction
  strd
@ CHECK: error: invalid instruction
  ldrd
@ CHECK: error: invalid instruction
  strd r0
@ CHECK: error: invalid instruction
  ldrd r0
@ CHECK: error: invalid instruction
  strd s0, [r0]
@ CHECK: error: invalid instruction
  ldrd s0, [r0]
