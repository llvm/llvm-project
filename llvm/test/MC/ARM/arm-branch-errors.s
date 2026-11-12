@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2>&1 | FileCheck %s

@------------------------------------------------------------------------------
@ Branch targets destined for ARM mode must == 0 (mod 4), otherwise (mod 2).
@------------------------------------------------------------------------------

        b #2
        bl #2
        beq #2

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK:        b #2
@ CHECK: note: invalid operand for instruction
@ CHECK: note: instruction requires: thumb
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK:        bl #2
@ CHECK: note: instruction requires: thumb
@ CHECK: note: invalid operand for instruction
@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK:        beq #2
@ CHECK: note: invalid operand for instruction
@ CHECK: note: instruction requires: thumb
