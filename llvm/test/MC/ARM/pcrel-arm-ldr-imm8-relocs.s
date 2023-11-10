@ RUN: llvm-mc -filetype=obj -triple=armv7 %s -o %t
@ RUN: llvm-readelf -r %t | FileCheck %s --check-prefix=ARM
@ RUN: llvm-objdump -d --triple=armv7 %t | FileCheck %s --check-prefix=ARM_ADDEND

@ ARM: R_ARM_LDRS_PC_G0
@ ARM: foo1
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: foo2
@ ARM: R_ARM_LDRS_PC_G0
@ ARM: foo3

// Value is decimal at the moment but hex in other cases (things could change)
@ ARM_ADDEND: r0, [pc, #-
@ ARM_ADDEND 8]
@ ARM_ADDEND: r0, [pc, #-
@ ARM_ADDEND 8]
@ ARM_ADDEND: r0, [pc, #-
@ ARM_ADDEND 8]

    .arm
    .section .text.bar, "ax"
    .balign 4
    .global bar
    .type bar, %function
bar:
    ldrh r0, foo1
    ldrsb r0, foo2
    ldrsh r0, foo3
    bx lr

    .section .data.foo, "a", %progbits
    .balign 4
    .global foo1
    .global foo2
    .global foo3
foo1:
foo2:
foo3:
    .word 0x11223344, 0x55667788
