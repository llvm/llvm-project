@ PR24346
@ RUN: not llvm-mc -triple=arm-linux-gnueabi -filetype=obj < %s 2>&1 | FileCheck %s

    .data
    .align 8
L2:
    .word 0
    .align 8
    .byte 0
L1:

    .text
@ CHECK: error: out of range immediate fixup value
    add r0, r0, #(L1 - L2)
