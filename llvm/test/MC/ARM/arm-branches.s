@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s

@------------------------------------------------------------------------------
@ Branch targets destined for ARM mode must == 0 (mod 4), otherwise (mod 2).
@------------------------------------------------------------------------------

        b #4
        bl #4
        beq #4
        blx #2

@ CHECK: b	#4                      @ encoding: [0x01,0x00,0x00,0xea]
@ CHECK: bl	#4                      @ encoding: [0x01,0x00,0x00,0xeb]
@ CHECK: beq	#4                      @ encoding: [0x01,0x00,0x00,0x0a]
@ CHECK: blx	#2                      @ encoding: [0x00,0x00,0x00,0xfb]
