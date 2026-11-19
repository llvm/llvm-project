# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec %t 2>&1 \
# RUN:              | FileCheck %s
#
# Check absolute symbols are created with a correct value, and that only
# external ones get default scope.
#
# CHECK: Creating graph symbols...
# CHECK:   6: Creating defined graph symbol for COFF symbol "abs" in (absolute) (index: -1)
# CHECK-NEXT:  0x53 (addressable + 0x00000000): size: 0x00000000, linkage: strong, scope: local, dead  -   abs
# CHECK:   7: Creating defined graph symbol for COFF symbol "ext_abs" in (absolute) (index: -1)
# CHECK-NEXT:  0x54 (addressable + 0x00000000): size: 0x00000000, linkage: strong, scope: default, dead  -   ext_abs

	.text
	.def	abs;
	.scl	3;
	.type	0;
	.endef
	.globl  abs
.set abs, 0x53

	.def	ext_abs;
	.scl	2;
	.type	0;
	.endef
	.globl  ext_abs
.set ext_abs, 0x54

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq
