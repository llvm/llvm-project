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

@------------------------------------------------------------------------------
@ Leading '$' on branch targets must not be dropped if part of symbol names
@------------------------------------------------------------------------------

        .global $foo
        .global $4
        b $foo
        bl $foo
        beq $foo
        blx $foo
        b $foo + 4
        bl $4
        beq $4 + 4

@ CHECK: b      ($foo)                      @ encoding: [A,A,A,0xea]
@ CHECK: bl     ($foo)                      @ encoding: [A,A,A,0xeb]
@ CHECK: beq    ($foo)                      @ encoding: [A,A,A,0x0a]
@ CHECK: blx    ($foo)                      @ encoding: [A,A,A,0xfa]
@ CHECK: b      #($foo)+4                   @ encoding: [A,A,A,0xea]
@ CHECK: bl     ($4)                        @ encoding: [A,A,A,0xeb]
@ CHECK: beq    #($4)+4                     @ encoding: [A,A,A,0x0a]

@------------------------------------------------------------------------------
@ Leading '$' should be allowed to introduce an expression
@------------------------------------------------------------------------------

        .global bar
        b $ 4
        bl $ bar + 4
        blx $ bar
@ CHECK: b	    #4                        @ encoding: [0x01,0x00,0x00,0xea]
@ CHECK: bl     #bar+4                    @ encoding: [A,A,A,0xeb]
@ CHECK: blx    bar                       @ encoding: [A,A,A,0xfa]
