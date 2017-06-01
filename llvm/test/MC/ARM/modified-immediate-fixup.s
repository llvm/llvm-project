@ PR24346
@ RUN: llvm-mc < %s -triple=arm-linux-gnueabi -filetype=obj -o - \
@ RUN: | llvm-objdump --disassemble -arch=arm - | FileCheck %s

    .data
    .align 8
L2:
    .word 0
    .align 8
    .word 0
L1:

    .text
@ CHECK: add r0, r0, #260
    add r0, r0, #(L1 - L2)
