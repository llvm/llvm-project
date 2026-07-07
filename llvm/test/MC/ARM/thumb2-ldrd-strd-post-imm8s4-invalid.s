@ RUN: not llvm-mc -triple=thumbv8 %s 2>&1 | FileCheck %s

.syntax unified
.thumb

@ CHECK: error: invalid instruction
@ CHECK:         ldrd r2, r3, [r4], #2
ldrd r2, r3, [r4], #2

@ CHECK: error: invalid instruction
@ CHECK:         ldrd r2, r3, [r4], #-2
ldrd r2, r3, [r4], #-2

@ CHECK: error: invalid instruction
@ CHECK:         strd r2, r3, [r4], #2
strd r2, r3, [r4], #2

@ CHECK: error: invalid instruction
@ CHECK:         strd r2, r3, [r4], #-2
strd r2, r3, [r4], #-2
