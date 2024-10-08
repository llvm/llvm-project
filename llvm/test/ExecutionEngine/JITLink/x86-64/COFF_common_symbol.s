# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check a common symbol is created.
#
# CHECK: Creating graph symbols...
# CHECK:      7: Creating defined graph symbol for COFF symbol "var" in (common) (index: 0)
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000004, linkage: weak, scope: default, dead  -   var

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4
main:
	movl	var(%rip), %eax
	retq

	.comm	var,4,2
