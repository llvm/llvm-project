@ RUN: not llvm-mc -triple=thumbv7-linux-gnueabi -mattr=+thumb2 %s -o /dev/null 2>&1 | FileCheck %s

.text
.thumb

ldrd r0, r1, [r2], #2
strd r0, r1, [r2], #-6

@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: ldrd r0, r1, [r2], #2
@ CHECK-NEXT:                    ^
@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: strd r0, r1, [r2], #-6
@ CHECK-NEXT:                    ^

ldrd r0, r1, [r2], #-1
strd r0, r1, [r2], #5

@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: ldrd r0, r1, [r2], #-1
@ CHECK-NEXT:                    ^
@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: strd r0, r1, [r2], #5
@ CHECK-NEXT:                    ^

ldrd r0, r1, [r2], #-15
strd r0, r1, [r2], #30

@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: ldrd r0, r1, [r2], #-15
@ CHECK-NEXT:                    ^
@ CHECK: note: invalid operand for instruction
@ CHECK-NEXT: strd r0, r1, [r2], #30
@ CHECK-NEXT:                    ^
