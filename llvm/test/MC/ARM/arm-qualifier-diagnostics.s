@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o - %s 2>&1 | FileCheck %s

	.syntax unified

	.type function,%function
function:
	ldr.n r0, [r0]

@ CHECK: error: instruction with .n (narrow) qualifier not allowed in arm mode
@ CHECK: 	ldr.n r0, [r0]
@ CHECK:           ^
@ CHECK-NOT: error: unexpected token in operand
@ CHECK-NOT: 	ldr.n r0, [r0]
@ CHECK-NOT:            ^

