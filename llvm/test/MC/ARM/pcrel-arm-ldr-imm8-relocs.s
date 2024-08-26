@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_ADDEND
@ RUN: llvm-mc -filetype=obj --triple=armebv7-unknown-unknown %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armebv7-unknown-unknown %t | FileCheck %s --check-prefix=ARM_ADDEND

@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: R_ARM_LDRS_PC_G0

// The value format is decimal in these specific cases, but it's hex for other
// ldr instructions. These checks are valid for both formats.

@ ARM_ADDEND: r0, [pc, #-{{(0x)?}}8]
@ ARM_ADDEND: r0, [pc, #-{{(0x)?}}8]
@ ARM_ADDEND: r0, [pc, #-{{(0x)?}}8]
@ ARM_ADDEND: r0, [pc, #-{{16|0x10}}]
@ ARM_ADDEND: r0, [pc, #-{{16|0x10}}]
@ ARM_ADDEND: r0, [pc]
@ ARM_ADDEND: r0, r1, [pc]

    .arm
    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldrh r0, foo
    ldrsb r0, foo
    ldrsh r0, foo
    ldrh r0, just_after-8
    ldrsb r0, just_after-8
    ldrsh r0, foo+8
    ldrd r0,r1, foo+8
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo
foo:
    .word 0x11223344, 0x55667788
just_after:
    .word 0x9900aabb, 0xccddeeff
