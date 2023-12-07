# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-pc-linux-gnu -filetype=obj \
# RUN:   -o %t/ELF_weak_defs_extra.o %S/Inputs/ELF_weak_defs_extra.s
# RUN: llvm-mc -triple x86_64-pc-linux-gnu -filetype=obj \
# RUN:   -o %t/ELF_weak_definitions.o %s
# RUN: llvm-jitlink -noexec -check=%s %t/ELF_weak_definitions.o \
# RUN:   %t/ELF_weak_defs_extra.o
#
# Check that objects linked separately agree on the address of weak symbols.
#
# jitlink-check: *{8}WeakDefAddrInThisFile = *{8}WeakDefAddrInExtraFile

	.text
	.file	"ELF_weak_definitions.c"
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
	movq	WeakDef@GOTPCREL(%rip), %rax
	movl	(%rax), %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

	.type	WeakDef,@object
	.data
	.weak	WeakDef
	.p2align	2
WeakDef:
	.long	1
	.size	WeakDef, 4

	.type	WeakDefAddrInThisFile,@object
	.globl	WeakDefAddrInThisFile
	.p2align	3
WeakDefAddrInThisFile:
	.quad	WeakDef
	.size	WeakDefAddrInThisFile, 8


	.type	extra_file_anchor,@object
	.globl	extra_file_anchor
	.p2align	3
extra_file_anchor:
	.quad	WeakDefAddrInExtraFile
	.size	extra_file_anchor, 8

	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym WeakDef
	.addrsig_sym WeakDefAddrInExtraFile
