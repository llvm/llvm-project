# RUN: rm -rf %t && mkdir -p %t
# RUN: yaml2obj %S/Inputs/COFF_strong_def.yaml -o %t/COFF_strong_1.o
# RUN: yaml2obj %S/Inputs/COFF_strong_def.yaml -o %t/COFF_strong_2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t/COFF_main.o
# RUN: 
# RUN: not llvm-jitlink -noexec %t/COFF_main.o %t/COFF_strong_1.o %t/COFF_strong_2.o \
# RUN: -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: -show-graph
#
# Check that duplicate strong definitions cause llvm-jitlink to terminate with error.
#

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