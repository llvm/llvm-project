@ RUN: not llvm-mc -triple=armv7-apple-darwin < %s 2>&1 | FileCheck %s

@------------------------------------------------------------------------------
@ Branch targets destined for ARM mode must == 0 (mod 4), otherwise (mod 2).
@------------------------------------------------------------------------------

        b #2
        bl #2
        beq #2

@ CHECK: error: instruction requires: thumb
@ CHECK:        b #2
@ CHECK: error: instruction requires: thumb
@ CHECK:        bl #2
@ CHECK: error: instruction requires: thumb
@ CHECK:        beq #2
