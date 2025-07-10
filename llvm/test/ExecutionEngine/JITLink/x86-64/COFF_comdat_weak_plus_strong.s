# FIXME: Comdat any + ordinary strong symbol should generate duplicate section error
# XFAIL: *
#
# RUN: rm -rf %t && mkdir -p %t
# RUN: yaml2obj %S/Inputs/COFF_comdat_weak_def.yaml -o %t/COFF_weak_1.o
# RUN: yaml2obj %S/Inputs/COFF_strong_def.yaml -o %t/COFF_strong.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t/COFF_main.o
#
# RUN: not llvm-jitlink -noexec %t/COFF_main.o %t/COFF_weak_1.o %t/COFF_strong.o \
# RUN:                  -slab-allocate 64Kb -slab-address 0xfff00000 \
# RUN:                  -slab-page-size 4096 -show-graphs=".*" 2>&1 | FileCheck %s
#
# Check that a combination of comdat any definition and strong definition
# generate duplicate definition error.
#
# CHECK: section strongfunc:
# CHECK-EMPTY:
# CHECK-NEXT:  block 0xfff0[[LO:[0-9a-f]+]] size = 0x00000001, align = 16, alignment-offset = 0
# CHECK-NEXT:    symbols:
# CHECK-NEXT:      0xfff0[[LO]] (block + 0x00000000): size: 0x00000001, linkage: strong, scope: default, live  -   func
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
