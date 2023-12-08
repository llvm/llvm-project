@ RUN: not llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERRORS %s

	.syntax unified
	.thumb

	.align 2
	.global constant_overflow
	.type constant_overflow,%function
constant_overflow:
	.inst.w 1 << 32
@ CHECK-ERRORS: inst.w operand is too big

