@ RUN: llvm-mc -filetype=obj -triple=thumbv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s --check-prefix=DISASM
@ RUN: llvm-mc -filetype=obj -triple=thumbv6m %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC

@ Thumb-1 `ldr r0, label` to a defined global label in the same section is
@ resolved at assembly time, keeping the 16-bit encoding (it was widened to
@ `ldr.w` with an R_ARM_THM_PC12, and rejected on thumbv6m).

@ RELOC: There are no relocations in this file.

@ DISASM: ldr     r0, [pc, #{{0x[0-9a-f]+}}]         @ {{0x[0-9a-f]+}} <pool>
@ DISASM: ldr     r1, [pc, #{{0x[0-9a-f]+}}]         @ {{0x[0-9a-f]+}} <pool>
@ DISASM: ldr     r2, [pc, #{{0x[0-9a-f]+}}]         @ {{0x[0-9a-f]+}} <pool>
@ DISASM-NOT: ldr.w

    .syntax unified
    .thumb
    .section .text.start, "ax"
    .balign 4
    .globl _start
_start:
    ldr   r0, pool
    ldr   r1, pool
    ldr   r2, pool
    bx    lr

    .balign 4
    .globl pool
pool:
    .word 0x11223344
