# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -abs var=0xcafef00d --debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check a default symbol is aliased as a weak external symbol.
#
# CHECK: Creating graph symbols...
# CHECK:      8: Creating defined graph symbol for COFF symbol ".weak.func.default.main" in .text (index: 1)
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000000, linkage: strong, scope: default, dead  -   .weak.func.default.main 
# CHECK-NEXT: 9: Creating defined graph symbol for COFF symbol "main" in .text (index: 1)
# CHECK-NEXT:   0x10 (block + 0x00000010): size: 0x00000000, linkage: strong, scope: default, dead  -   main
# CHECK-NEXT: 6: Creating weak external symbol for COFF symbol "func" in section 0
# CHECK-NEXT:   0x0 (block + 0x00000000): size: 0x00000000, linkage: weak, scope: default, dead  -   func

	.text

	.def	func;
	.scl	2;
	.type	32;
	.endef
	.weak	func
	.p2align	4, 0x90
func:
	retq

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq
