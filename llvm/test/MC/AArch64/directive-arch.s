// RUN: llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch armv8-a+crypto

	aesd v0.16b, v2.16b

# CHECK: 	aesd	v0.16b, v2.16b

	.arch armv8.1-a+ras
	esb

# CHECK: 	esb

