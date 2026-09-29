@ RUN: not llvm-mc -triple thumbv7-apple-ios -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s
        @ Size: 2 bytes
        .thumb_func _f1
        .thumb
        .globl _f1
_f1:
        bx lr

        @ A misaligned ARM destination.
        .arm
        .byte 0x0
        .globl _misaligned
_misaligned:
        bx lr

        @ And a properly aligned one.
        .globl _aligned
        .p2align 2
        .arm
_aligned:
        bx lr

        @ Align this Thumb function so we can predict the outcome of
        @ "Align(PC, 4)" during blx operation.
        .thumb_func _test
        .thumb
        .p2align 2
        .globl _test
_test:
        blx _misaligned       @ PC=0 (mod 4)
        movs r0, r0
        blx _misaligned       @ PC=2 (mod 4)
        movs r0, r0
        blx _aligned          @ PC=0 (mod 4)
        movs r0, r0
        blx _aligned          @ PC=2 (mod 4)

@ CHECK: error: misaligned ARM call destination
@ CHECK:   blx _misaligned
@ CHECK: error: misaligned ARM call destination
@ CHECK:   blx _misaligned
