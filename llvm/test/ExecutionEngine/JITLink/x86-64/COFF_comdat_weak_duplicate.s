# RUN: rm -rf %t && mkdir -p %t
# RUN: yaml2obj %S/Inputs/COFF_comdat_weak_def.yaml -o %t/COFF_weak_1.o
# RUN: yaml2obj %S/Inputs/COFF_comdat_weak_def.yaml -o %t/COFF_weak_2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t/COFF_main.o
# RUN:
# RUN: llvm-jitlink -noexec %t/COFF_main.o %t/COFF_weak_1.o %t/COFF_weak_2.o \
# RUN: -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: -show-graphs='.*' -noexec 2>&1 | FileCheck %s
#
# Check that duplicate comdat any definitions don't generate duplicate definition error.
#
# CHECK: section weakfunc:
# CHECK-EMPTY:
# CHECK-NEXT:  block 0xfff01000 size = 0x00000001, align = 16, alignment-offset = 0
# CHECK-NEXT:    symbols:
# CHECK-NEXT:      0xfff01000 (block + 0x00000000): size: 0x00000001, linkage: weak, scope: default, live  -   func
# CHECK-NEXT:    no edges

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
    callq func
	retq
