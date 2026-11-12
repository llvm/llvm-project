@ RUN: not llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.syntax unified
	.arm

	.align 2
	.global constant_overflow
	.type constant_overflow,%function
constant_overflow:
	.inst 1 << 32
@ CHECK-ERROR: inst operand is too big


