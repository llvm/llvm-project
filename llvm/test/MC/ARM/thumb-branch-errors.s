@ RUN: not llvm-mc -triple=thumbv7-apple-darwin < %s 2>&1 | FileCheck %s

@------------------------------------------------------------------------------
@ Branch targets destined for ARM mode must == 0 (mod 4), otherwise (mod 2).
@------------------------------------------------------------------------------

        b #1
        bl #1
        cbnz r2, #1
        beq #1
        blx #2

@ CHECK: error: branch target out of range
@ CHECK:         b #1
@ CHECK: error: invalid operand for instruction
@ CHECK:         bl #1
@ CHECK: error: invalid operand for instruction
@ CHECK:         cbnz r2, #1
@ CHECK: error: branch target out of range
@ CHECK:         beq #1
@ CHECK: error: invalid operand for instruction
@ CHECK:         blx #2
