@ RUN: llvm-mc -filetype=obj -triple=thumbv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s --check-prefix=THUMB_ADDEND

@ All the ldr variants produce a relocation
@ THUMB: R_ARM_THM_PC12
@ THUMB: foo3
@ THUMB: R_ARM_THM_PC12
@ THUMB: foo4
@ THUMB: R_ARM_THM_PC12
@ THUMB: foo5

@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]

    .thumb
    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldrh r0, foo3
    ldrsb r0, foo4
    ldrsh r0, foo5
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo3
    .global foo4
    .global foo5
foo3:
foo4:
foo5:
    .word 0x11223344, 0x55667788
