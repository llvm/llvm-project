@ RUN: not llvm-mc -triple armv7a--none-eabi < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv7a--none-eabi < %s 2>&1 | FileCheck %s

  msr apsr_c, r0
@ CHECK: invalid operand for instruction
  msr cpsr_w, r0
@ CHECK: invalid operand for instruction
  msr cpsr_cc, r0
@ CHECK: invalid operand for instruction
  msr xpsr_c, r0
@ CHECK: invalid operand for instruction
