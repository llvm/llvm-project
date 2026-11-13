@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=DISASM
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC

@ Same-section `ldr`/`adr` to a defined global label is resolved at assembly
@ time; relocations are kept for weak, undefined, ifunc and cross-section
@ symbols and for branches.

@ RELOC: R_ARM_LDR_PC_G0 {{[0-9a-f]+ }}weak_pool
@ RELOC: R_ARM_LDR_PC_G0 {{[0-9a-f]+ }}undef_sym
@ RELOC: R_ARM_LDR_PC_G0 {{[0-9a-f]+ }}ifunc_pool
@ RELOC: R_ARM_LDR_PC_G0 {{[0-9a-f]+ }}cross_pool
@ RELOC: R_ARM_JUMP24 {{[0-9a-f]+ }}pool
@ RELOC-NOT: R_ARM

@ DISASM-LABEL: <_start>:
@ DISASM: ldr r0, [pc, #0x28] @ 0x30 <pool>
@ DISASM: ldr r1, [pc, #0x28] @ 0x34 <pool+0x4>
@ DISASM: ldrh r6, [pc, #16] @ 0x30 <pool>
@ DISASM: ldrd r8, r9, [pc, #12] @ 0x30 <pool>
@ DISASM: vldr d0, [pc, #8] @ 0x30 <pool>
@ DISASM: add r9, pc, #4

    .syntax unified
    .arm
    .section .text.start, "ax"
    .balign 4
    .globl _start
    .type _start, %function
_start:
    ldr   r0, pool          @ resolved
    ldr   r1, pool+4        @ resolved
    ldr   r2, weak_pool     @ relocation
    ldr   r3, undef_sym     @ relocation
    ldr   r4, ifunc_pool    @ relocation
    ldr   r5, cross_pool    @ relocation
    ldrh  r6, pool          @ resolved
    ldrd  r8, r9, pool      @ resolved
    vldr  d0, pool          @ resolved
    adr   r9, pool          @ resolved
    b     pool              @ relocation (branch)
    bx    lr

    .balign 4
    .globl pool
pool:
    .word 0x11223344
    .word 0x55667788

    .weak weak_pool
weak_pool:
    .word 0x99aabbcc

    .type ifunc_pool, %gnu_indirect_function
    .globl ifunc_pool
ifunc_pool:
    bx lr

    .section .data.pool, "a", %progbits
    .balign 4
    .globl cross_pool
cross_pool:
    .word 0xddeeff00
