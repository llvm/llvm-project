# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -check=%s %t.o

# jitlink-check: decode_operand(main, 4) = _GLOBAL_OFFSET_TABLE_ - next_pc(main)

	.text
	.section	.text.main,"ax",@progbits
	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
	leal    _GLOBAL_OFFSET_TABLE_(%rip), %ebx
	xorl	%eax, %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main

