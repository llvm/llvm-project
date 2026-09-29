@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ADDEND

@ RUN: llvm-mc -filetype=obj --triple=armebv7-unknown-unknown %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=RELOC
@ RUN: llvm-objdump -d --triple=armebv7-unknown-unknown %t | FileCheck %s --check-prefix=ADDEND

    .section .text._func1, "ax"

    .balign 4
    .global _func1
    .type _func1, %function
_func1:
    adr r0, _func2
@ RELOC: R_ARM_ALU_PC_G0
    .thumb
    adr r0, _func2
@ RELOC: R_ARM_THM_ALU_PREL_11_0
    bx lr

@ ADDEND:      sub	r0, pc, #8
@ ADDEND-NEXT: adr.w	r0, #-4

