@ RUN: not llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.syntax unified
	.thumb

	.align 2
	.global suffixes_required_in_thumb
	.type suffixes_required_in_thumb,%function
suffixes_required_in_thumb:
	.inst 0xff00
@ CHECK-ERROR: cannot determine Thumb instruction size, use inst.n/inst.w instead

