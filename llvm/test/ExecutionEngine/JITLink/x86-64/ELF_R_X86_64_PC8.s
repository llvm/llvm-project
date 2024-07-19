# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o
#
# Check R_X86_64_PC8 handling.

	.text
	.globl	main
	.type	main,@function
main:
	xorl	%eax, %eax
	retq
	.size	main, .-main

	.type	P,@object
	.globl	P
P:
	.byte main-. # Generate R_X86_64_PC8 relocation.
  .size P, .-P
