// RUN: llvm-mc -triple aarch64-unknown-none-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.arch armv8-a+crypto

	aesd v0.16b, v2.16b
	eor v0.16b, v0.16b, v2.16b

# CHECK: 	aesd	v0.16b, v2.16b
# CHECK:        eor     v0.16b, v0.16b, v2.16b

