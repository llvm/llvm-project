# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check absolute symbol is created with a correct value.
#
# CHECK: Creating graph symbols...
# CHECK:   6: Creating defined graph symbol for COFF symbol "abs" in (absolute) (index: -1)
# CHECK-NEXT:  0x53 (addressable + 0x00000000): size: 0x00000000, linkage: strong, scope: local, dead  -   abs

	.text
	.def	abs;
	.scl	3;
	.type	0;
	.endef
	.globl  abs
.set abs, 0x53

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq
