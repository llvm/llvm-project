@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s

	.cpu armv7

	.type double_diagnostics,%function
double_diagnostics:
	.word invalid(invalid) + 32

@ CHECK: error: invalid variant 'invalid'
@ CHECK-NOT: error: unexpected token at start of statement

