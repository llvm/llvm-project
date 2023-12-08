// RUN: llvm-mc -triple x86_64-apple-darwin -x86-asm-syntax=intel %s | FileCheck %s
// rdar://14961158
	.text
	.align 16
	.globl FUNCTION_NAME
	.private_extern	FUNCTION_NAME
FUNCTION_NAME:
	.intel_syntax
	cmp rdi, 1
	jge 1f
// CHECK:	jge	Ltmp0
	add rdi, 2
// CHECK: addq $2, %rdi
1:
// CHECK:	Ltmp0:
	add rdi, 1
	ret
