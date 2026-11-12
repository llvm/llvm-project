@ RUN: not llvm-mc -triple armv7 -filetype asm -o /dev/null %s 2>&1 \
@ RUN:     | FileCheck %s -strict-whitespace

	.text
	.thumb

	.fpu invalid
@ CHECK: error: Unknown FPU name
@ CHECK: .fpu invalid
@ CHECK:      ^
