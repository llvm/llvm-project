# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -alias x=y %t.o
#
# Check that the -alias option works.

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	4, 0x90
_main:
	movq	x@GOTPCREL(%rip), %rax
	movl	(%rax), %eax
	retq

	.section	__DATA,__data
	.globl	y
	.p2align	2
y:
	.long	42

.subsections_via_symbols
