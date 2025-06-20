# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o
#
# Check R_X86_64_PC* handling.

	.text
	.globl	main
	.type	main,@function
main:
	xorl	%eax, %eax
	retq
	.size	main, .-main

	.rodata
	.byte	main-. # Generate R_X86_64_PC8 relocation.
	.short	main-. # Generate R_X86_64_PC16 relocation.
	.long	main-. # Generate R_X86_64_PC32 relocation.
	.quad	main-. # Generate R_X86_64_PC64 relocation.
