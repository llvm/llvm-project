# RUN: rm -rf %t && mkdir -p %t
# RUN: yaml2obj %S/Inputs/COFF_weak_nolibrary_serach_def.yaml -o %t/COFF_weak.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t/COFF_main.o
# RUN: not llvm-jitlink -noexec -abs __ImageBase=0xfff00000 \
# RUN: -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: %t/COFF_weak.o %t/COFF_main.o 2>&1 | FileCheck %s
#
# Check weak external symbol with IMAGE_WEAK_EXTERN_SEARCH_NOLIBRARY Characteristics
# does not export symbol.
#
# CHECK: llvm-jitlink error: Symbols not found: [ func ]

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