# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check a COMDAT any symbol is exported as a weak symbol.
#
# CHECK: Creating graph symbols...
# CHECK:      6: Creating defined graph symbol for COFF symbol ".text" in .text (index: 4)
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000001, linkage: strong, scope: local, dead  -   <anonymous symbol>
# CHECK-NEXT: 8: Exporting COMDAT graph symbol for COFF symbol "func" in section 4
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000001, linkage: weak, scope: default, dead  -   func

	.text

	.def	func;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,func
	.globl	func
	.p2align	4, 0x90
func:
	retq

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	main
	.p2align	4, 0x90
main:
	retq
