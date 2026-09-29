@ RUN: not llvm-mc < %s -triple armv4-unknown-unknown -show-encoding 2>&1 | FileCheck %s

@ PR18524
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: instruction requires: armv5t
@ CHECK: note: instruction requires: thumb2
clz r4,r9

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: instruction requires: armv6t2
@ CHECK: note: instruction requires: thumb2
rbit r4,r9

@ CHECK: error: instruction requires: armv6t2
movw r4,#0x1234

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: invalid operand for instruction
@ CHECK: note: operand must be a register in range [r0, r15]
@ CHECK: note: instruction requires: armv6t2
mov  r4,#0x1234
