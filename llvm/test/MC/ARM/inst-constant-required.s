@ RUN: not llvm-mc %s -triple=armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.syntax unified
	.arm

	.align 2
	.global constant_expression_required
	.type constant_expression_required,%function
constant_expression_required:
.Label:
	movs r0, r0
	.inst .Label
@ CHECK-ERROR: expected constant expression

