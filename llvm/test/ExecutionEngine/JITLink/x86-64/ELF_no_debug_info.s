# REQUIRES: native && x86_64-linux

# RUN: rm -rf %t && mkdir %t
# RUN: llvm-mc -triple=x86_64-unknown-linux \
# RUN:     -filetype=obj -o %t/ELF_x86-64_no_debug_info.o %s
# RUN: llvm-jitlink %t/ELF_x86-64_no_debug_info.o

# Check if everything works in the absent of any debug information.

	.text
	.globl	main                            # -- Begin function main
	.p2align	4
	.type	main,@function
main:                                   # @main
	pushq	%rbp
	movq	%rsp, %rbp
	movl	$0, -4(%rbp)
	movl	$0, %eax
	popq	%rbp
	retq
