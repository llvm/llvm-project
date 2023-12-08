@ RUN: llvm-mc -triple thumbv7-apple-ios7.0 -filetype=obj -o - %s | \
@ RUN: llvm-readobj -r - | FileCheck %s

        @ MachO relocations that end up expressed as internal
        @ (scattered) still need to have the type set correctly.

        .text
        .thumb_func
        .thumb
        .globl _with_thumb
_with_thumb:
        bl _dest+10
        blx _dest+20

        .globl _with_arm
        .arm
_with_arm:
        bl _dest+10
        blx _dest+20
        bne _dest+30
        b _dest+40

        .data
_dest:
        .word 42

@ CHECK: Relocations [
@ CHECK-NEXT: Section __text {
@ CHECK-NEXT: 0x14 1 2 n/a ARM_RELOC_BR24 1 0x18
@ CHECK-NEXT: 0x10 1 2 n/a ARM_RELOC_BR24 1 0x18
@ CHECK-NEXT: 0xC 1 2 n/a ARM_RELOC_BR24 1 0x18
@ CHECK-NEXT: 0x8 1 2 n/a ARM_RELOC_BR24 1 0x18
@ CHECK-NEXT: 0x4 1 2 n/a ARM_THUMB_RELOC_BR22 1 0x18
@ CHECK-NEXT: 0x0 1 2 n/a ARM_THUMB_RELOC_BR22 1 0x18
