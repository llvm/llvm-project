@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
        .syntax unified
bar:
@ The line is duplicated on purpose, it is legal to redefine a req with
@ the same value.
fred .req r5
fred .req r5
        mov r11, fred
.unreq fred
fred .req r6
        mov r1, fred

@ CHECK: mov	r11, r5                 @ encoding: [0x05,0xb0,0xa0,0xe1]
@ CHECK: mov	r1, r6                  @ encoding: [0x06,0x10,0xa0,0xe1]
