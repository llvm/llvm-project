@ RUN: not llvm-mc -triple armv7a-linux-gnueabihf %s -filetype=obj -o %t.o 2>&1 | FileCheck %s

@ The relaxations should be applied using the subtarget from the fragment
@ containing the fixup and not the per module subtarget.

        .syntax unified
        .thumb
        @ Place a literal pool out of range of the 16-bit ldr but within
        @ range of the 32-bit ldr.w
        .text
        @ Relaxation to ldr.w as target triple supports Thumb2
        ldr r0,=0x12345678
        .arch armv4t
        @ No relaxation as v4t does not support Thumb
        @ expect out of range error message
        ldr r0,=0x87654321
        .space 1024

@ CHECK: error: out of range pc-relative fixup value
@ CHECK-NEXT: ldr r0,=0x87654321
