# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -entry hook %t.o
#
# Make sure that an empty __text section doesn't cause any problems.

  .section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 15
l_empty:

	.section	__TEXT,__const
	.globl	hook
	.p2align	2
hook:
	.long	42

.subsections_via_symbols
