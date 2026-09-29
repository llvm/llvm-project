@ RUN: not llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - 2>&1 \
@ RUN:   | FileCheck -check-prefix CHECK-ERROR %s

	.syntax unified
	.arm

	.align 2
	.global suffixes_invalid_in_arm
	.type suffixes_invalid_in_arm,%function
suffixes_invalid_in_arm:
	.inst.n 2
@ CHECK-ERROR: width suffixes are invalid in ARM mode
	.inst.w 4
@ CHECK-ERROR: width suffixes are invalid in ARM mode

