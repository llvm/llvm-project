# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-pc-linux -filetype=obj -o %t/tmp.o %s
# RUN: llvm-rtdyld -triple=x86_64-pc-linux -verify %t/tmp.o

	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   #
	.cfi_startproc
# %bb.0:
	movl	o42, %eax
	retq
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        # -- End function
	.type	o1,@object              #
	.comm	o1,4,4
	.type	o2,@object              #
	.comm	o2,4,4
	.type	o3,@object              #
	.comm	o3,4,4
	.type	o4,@object              #
	.comm	o4,4,4
	.type	o5,@object              #
	.comm	o5,4,4
	.type	o6,@object              #
	.comm	o6,4,4
	.type	o7,@object              #
	.comm	o7,4,4
	.type	o8,@object              #
	.comm	o8,4,4
	.type	o9,@object              #
	.comm	o9,4,4
	.type	o10,@object             #
	.comm	o10,4,4
	.type	o11,@object             #
	.comm	o11,4,4
	.type	o12,@object             #
	.comm	o12,4,4
	.type	o13,@object             #
	.comm	o13,4,4
	.type	o14,@object             #
	.comm	o14,4,4
	.type	o15,@object             #
	.comm	o15,4,4
	.type	o16,@object             #
	.comm	o16,4,4
	.type	o17,@object             #
	.comm	o17,4,4
	.type	o18,@object             #
	.comm	o18,4,4
	.type	o19,@object             #
	.comm	o19,4,4
	.type	o20,@object             #
	.comm	o20,4,4
	.type	o21,@object             #
	.comm	o21,4,4
	.type	o22,@object             #
	.comm	o22,4,4
	.type	o23,@object             #
	.comm	o23,4,4
	.type	o24,@object             #
	.comm	o24,4,4
	.type	o25,@object             #
	.comm	o25,4,4
	.type	o26,@object             #
	.comm	o26,4,4
	.type	o27,@object             #
	.comm	o27,4,4
	.type	o28,@object             #
	.comm	o28,4,4
	.type	o29,@object             #
	.comm	o29,4,4
	.type	o30,@object             #
	.comm	o30,4,4
	.type	o31,@object             #
	.comm	o31,4,4
	.type	o32,@object             #
	.comm	o32,4,4
	.type	o33,@object             #
	.comm	o33,4,4
	.type	o34,@object             #
	.comm	o34,4,4
	.type	o35,@object             #
	.comm	o35,4,4
	.type	o36,@object             #
	.comm	o36,4,4
	.type	o37,@object             #
	.comm	o37,4,4
	.type	o38,@object             #
	.comm	o38,4,4
	.type	o39,@object             #
	.comm	o39,4,4
	.type	o40,@object             #
	.comm	o40,4,4
	.type	o41,@object             #
	.comm	o41,4,4
	.type	o42,@object             #
	.comm	o42,4,4

	.section	".note.GNU-stack","",@progbits
