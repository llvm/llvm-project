@ RUN: llvm-mc -triple=armv7-linux-gnueabi -filetype=obj < %s | llvm-objdump -t - | FileCheck %s

    .text
@ $a at 0x0000
    add r0, r0, r0
@ $d at 0x0004
    .word 42
    .thumb
@ $t at 0x0008
    adds r0, r0, r0
    adds r0, r0, r0
@ $a at 0x000c
    .arm
    add r0, r0, r0
@ $t at 0x0010
    .thumb
    adds r0, r0, r0
@ $d at 0x0012
    .ascii "012"
    .byte 1
    .byte 2
    .byte 3
@ $a at 0x0018
    .arm
    add r0, r0, r0

@ CHECK:      00000000 l       .text  00000000 $a
@ CHECK-NEXT: 00000004 l       .text  00000000 $d
@ CHECK-NEXT: 00000008 l       .text  00000000 $t
@ CHECK-NEXT: 0000000c l       .text  00000000 $a
@ CHECK-NEXT: 00000010 l       .text  00000000 $t
@ CHECK-NEXT: 00000012 l       .text  00000000 $d
@ CHECK-NEXT: 00000018 l       .text  00000000 $a
