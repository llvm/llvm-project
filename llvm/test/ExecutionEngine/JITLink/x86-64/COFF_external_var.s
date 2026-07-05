# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: not llvm-jitlink -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN: -abs var=0x7fff00000000 -noexec %t
#
# Check data access to a external variable out of reach causes an error.
#

	.text

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main 
	.p2align	4, 0x90
main:
	movl	var(%rip), %eax
	retq
