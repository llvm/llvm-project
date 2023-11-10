@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_ADDEND
@ RUN: llvm-mc -filetype=obj -triple=thumbv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s --check-prefix=THUMB_ADDEND

@ ARM: R_ARM_LDR_PC_G0
@ ARM: foo1
@ ARM: R_ARM_LDR_PC_G0
@ ARM: foo2

@ ARM_ADDEND: r0, [pc, #-0x8]
@ ARM_ADDEND: r0, [pc, #-0x8]
@ ARM_ADDEND: r0, [pc, #-0x10]

@ THUMB: R_ARM_THM_PC12
@ THUMB: foo1
@ THUMB: R_ARM_THM_PC12
@ THUMB: foo2

@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0xc]

    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldr r0, foo1
    ldrb r0, foo2
    ldr r0, foo3-8
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo1
    .global foo2
    .global foo3
foo1:
foo2:
    .word 0x11223344, 0x55667788
foo3:
    .word 0x99aabbcc, 0xddeeff00
