// Regression test for a bug in which --disassemble-all had the side effect
// of stopping mapping symbols from being checked in code sections, so that
// mixed Arm/Thumb code would not all be correctly disassembled.

@ RUN: llvm-mc -triple arm-unknown-linux -filetype=obj %s -o %t.o
@ RUN: llvm-objdump --no-print-imm-hex -d %t.o | FileCheck %s
@ RUN: llvm-objdump --no-print-imm-hex -d --disassemble-all %t.o | FileCheck %s

@ CHECK:       00000000 <armfunc>:
@ CHECK-NEXT:        0: e2800001      add     r0, r0, #1
@ CHECK-NEXT:        4: e12fff1e      bx      lr
@ CHECK-NEXT:        8: 00 00         .short  0x0000
@ CHECK-EMPTY:
@ CHECK:       0000000a <thmfunc>:
@ CHECK-NEXT:        a: f100 0001     add.w   r0, r0, #1
@ CHECK-NEXT:        e: 4770          bx      lr
@ CHECK-NEXT:       10: 00 00         .short  0x0000

	.arch armv8a
        .text

        .arm
	.global	armfunc
	.type	armfunc, %function
armfunc:
        add     r0, r0, #1
        bx      lr
        @@ Test that this is not displayed as a .word
        .space  2

        .thumb
	.global	thmfunc
	.type	thmfunc, %function
thmfunc:
        add     r0, r0, #1
        bx      lr
        .space  2
