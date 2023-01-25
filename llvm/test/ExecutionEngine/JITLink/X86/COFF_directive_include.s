# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: not llvm-jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check an external symbol "foo" is generated and not dead-stripped
# because of include directive which turned into symbol not found error.
#
# CHECK: error: Symbols not found: [ foo ]

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq

	.section .drectve,"yn"
	.ascii "/include:foo"
