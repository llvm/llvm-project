@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_ADDEND
@ RUN: llvm-mc -filetype=obj -triple=thumbv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s --check-prefix=THUMB_ADDEND
@ RUN: llvm-mc -filetype=obj -triple=armebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armebv7 %t | FileCheck %s --check-prefix=ARM_ADDEND
@ RUN: llvm-mc -filetype=obj -triple=thumbebv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbebv7 %t | FileCheck %s --check-prefix=THUMB_ADDEND

@ ARM: R_ARM_LDR_PC_G0
@ ARM: R_ARM_LDR_PC_G0
@ ARM: R_ARM_LDR_PC_G0
@ ARM: R_ARM_LDR_PC_G0
@ ARM_ADDEND: r0, [pc, #-0x8]
@ ARM_ADDEND: r0, [pc, #-0x8]
@ ARM_ADDEND: r0, [pc, #-0x10]
@ ARM_ADDEND: r0, [pc]

@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0xc]
@ THUMB_ADDEND: r0, [pc, #0x4]

    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldr r0, foo1
    ldrb r0, foo1
    ldr r0, foo2-8
    ldrb r0, foo1+8
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo1
    .global foo2
foo1:
    .word 0x11223344, 0x55667788
foo2:
    .word 0x99aabbcc, 0xddeeff00
