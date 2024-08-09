# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=aarch64-windows-msvc %s -o %t
# RUN: llvm-jitlink --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check function  symbol is created with a correct value.
#
# CHECK: Creating graph symbols...
# CHECK: 6: Creating defined graph symbol for COFF symbol "foo" in .text (index: 1)
# CHECK-NEXT: 0x0 (block + 0x00000000): size: 0x00000000, linkage: strong, scope: default, dead  -   foo
#
# CHECK: Processing relocations:
# CHECK: .text:
# CHECK-NEXT:  edge@0x4: 0x0 + 0x4 -- Branch19 -> foo + -622592

    .text
	.def	foo;
	.scl	2;
	.type	32;
	.endef
	.globl	foo
	.p2align	2
foo:
	ret

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	2
main:
	cbz xzr, foo
	ret