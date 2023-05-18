@ RUN: not llvm-mc -triple=armv7-linux-gnueabi %s 2>&1 | FileCheck %s
.text
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK:         ldrd    r0, [r0, #512]
@ CHECK: note: invalid operand for instruction
@ CHECK: note: instruction requires: thumb2
        ldrd    r0, [r0, #512]

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK:         strd    r0, [r0, #512]
@ CHECK: note: invalid operand for instruction
@ CHECK: note: instruction requires: thumb2
        strd    r0, [r0, #512]
