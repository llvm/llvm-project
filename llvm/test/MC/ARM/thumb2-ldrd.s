@ RUN: not llvm-mc -triple thumb-eabi -mattr=+thumb2 %s -o /dev/null 2>&1 \
@ RUN:	  | FileCheck %s

@ rdar://14479780

ldrd r0, r0, [pc, #0]
ldrd r0, r0, [r1, #4]
ldrd r0, r0, [r1], #4
ldrd r0, r0, [r1, #4]!

@ CHECK: error: destination operands can't be identical
@ CHECK: error: destination operands can't be identical
@ CHECK: error: destination operands can't be identical
@ CHECK: error: destination operands can't be identical
@ CHECK-NOT: error: destination operands can't be identical

