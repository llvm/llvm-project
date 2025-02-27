@ RUN: llvm-mc -triple armv7-unknown-linux-gnueabi -filetype obj -o - %s \
@ RUN:   | llvm-objdump -d -z - | FileCheck %s

        ldr r0, =0x01020304
@ CHECK: ldr
        .ltorg
@ CHECK: 0x01020304
        ldr r0, =0x01020304
        ldr r0, =0x01020304
        ldr r0, =0x01020304
@ CHECK: ldr
@ CHECK: ldr
@ CHECK: ldr
        .ltorg
@ CHECK: 0x01020304
    .rep 1028
        .word 0
    .endr
@ CHECK: 0x00000000

        ldr r0, =0x01020304
@ CHECK: ldr
        .ltorg
@ CHECK: 0x01020304
    .rep 1028
        .word 0
    .endr
