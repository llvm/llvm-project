@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text
.thumb

@ CHECK: error: operand must be a register in range [r0, r12] or r14
@ CHECK: error: operand must be a register in range [r0, r12] or r14
@ CHECK: error: operand must be a register in range [r0, r12] or r14
strd r12, SP, [r0, #256]
strd r12, SP, [r0, #256]!
strd r12, SP, [r0], #256
