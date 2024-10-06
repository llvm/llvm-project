# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=aarch64-windows-msvc %s -o %t
# RUN: llvm-jitlink -debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check function  symbol is created with a correct value.
#
# CHECK: Creating graph symbols...
# CHECK: 7: Creating defined graph symbol for COFF symbol "retv" in .bss (index: 3)
# CHECK-NEXT: 0x0 (block + 0x00000000): size: 0x00000000, linkage: strong, scope: default, dead  -   retv
#
# CHECK: Processing relocations:
# CHECK: .text:
# CHECK-NEXT: edge@0x0: 0x0 + 0x0 -- PageBase_Rel21 -> retv + -1879048184
# CHECK-NEXT: edge@0x4: 0x0 + 0x4 -- PageOffset_12L -> retv + -1186987776

	.text
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	2
main:
	adrp	x8, retv
	ldr	w0, [x8, :lo12:retv]
	ret

	.bss
	.globl	retv
	.p2align	2, 0x0
retv:
	.word	0