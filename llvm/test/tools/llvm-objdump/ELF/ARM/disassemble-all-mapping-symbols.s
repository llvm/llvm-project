// Regression test for a bug in which --disassemble-all had the side effect
// of stopping mapping symbols from being checked in code sections, so that
// mixed Arm/Thumb code would not all be correctly disassembled.

@ RUN: llvm-mc -triple arm-unknown-linux -filetype=obj %s -o %t.o
@ RUN: llvm-objdump --no-print-imm-hex -d %t.o | FileCheck %s
@ RUN: llvm-objdump --no-print-imm-hex -d --disassemble-all %t.o | FileCheck %s

@ CHECK: 00000000 <armfunc>:
@ CHECK:        0: e2800001      add     r0, r0, #1
@ CHECK:        4: e12fff1e      bx      lr
@
@ CHECK: 00000008 <thmfunc>:
@ CHECK:        8: f100 0001     add.w   r0, r0, #1
@ CHECK:        c: 4770          bx      lr

	.arch armv8a
        .text

        .arm
	.global	armfunc
	.type	armfunc, %function
armfunc:
        add     r0, r0, #1
        bx      lr

        .thumb
	.global	thmfunc
	.type	thmfunc, %function
thmfunc:
        add     r0, r0, #1
        bx      lr
