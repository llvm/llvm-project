@ RUN: not llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.syntax unified
	.thumb

	.align 2
	.global constant_overflow
	.type constant_overflow,%function
constant_overflow:
	.inst.n 1 << 31
@ CHECK-ERROR: inst.n operand is too big, use inst.w instead

