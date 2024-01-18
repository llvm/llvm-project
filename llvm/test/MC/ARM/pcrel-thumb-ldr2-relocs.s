@ RUN: llvm-mc -filetype=obj -triple=thumbv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbv7 %t | FileCheck %s --check-prefix=THUMB_ADDEND
@ RUN: llvm-mc -filetype=obj --triple=thumbebv7-unknown-unknown %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-objdump -d --triple=thumbebv7-unknown-unknown %t | FileCheck %s --check-prefix=THUMB_ADDEND

@ All the ldr variants produce a relocation
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12
@ THUMB: R_ARM_THM_PC12

@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #-0x4]
@ THUMB_ADDEND: r0, [pc, #0x4]
@ THUMB_ADDEND: r0, [pc, #-0xc]
@ THUMB_ADDEND: r0, [pc, #0x4]

    .thumb
    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldrh r0, foo1
    ldrsb r0, foo1
    ldrsh r0, foo1
    ldrh r0, foo1+8
    ldrsb r0, foo2-8
    ldrsh r0, foo1+8
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo1
    .global foo2
foo1:
    .word 0x11223344, 0x55667788
foo2:
    .word 0x9900aabb, 0xccddeeff
