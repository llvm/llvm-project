@ RUN: llvm-mc -triple thumbv7-apple-ios -filetype=obj %s -o %t
@ RUN: llvm-objdump --macho -d %t | FileCheck %s

        @ Size: 2 bytes
        .thumb_func _f1
        .thumb
        .globl _f1
_f1:
        bx lr

        @ A properly aligned ARM function
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
        blx _elsewhere
        blx _aligned    @ PC=0 (mod 4)
        blx _aligned    @ PC=0 (mod 4)
        movs r0, r0
        blx _aligned    @ PC=2 (mod 4)
        blx _f1

@ CHECK: blx _elsewhere
@ CHECK: ff f7 fa ef blx _aligned
@ CHECK: ff f7 f8 ef blx _aligned
@ CHECK: ff f7 f6 ef blx _aligned
@ CHECK: ff f7 f2 ef blx _f1
