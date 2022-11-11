# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: not llvm-jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check object without alternatename directive fails because of
# external symbol not found error.
#
# CHECK: error: Symbols not found: [ foo ]
	.text
	
	.def	 foo_def;
	.scl	2;
	.type	32;
	.endef
	.globl	foo_def
	.p2align	4, 0x90
foo_def:
	retq

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	callq foo
	retq
