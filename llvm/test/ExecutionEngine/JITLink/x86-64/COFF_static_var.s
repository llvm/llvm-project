# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -abs var=0xcafef00d --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check a local symbol is created for a static variable.
#
# CHECK: Creating graph symbols...
# CHECK:      7: Creating defined graph symbol for COFF symbol "var" in .data (index: 2)
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000000, linkage: strong, scope: local, dead  -   var

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq

	.data
	.p2align	2
var:
	.long	53
