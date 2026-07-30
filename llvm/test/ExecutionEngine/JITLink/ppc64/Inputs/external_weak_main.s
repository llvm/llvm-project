	.text
	.abiversion 2
	.weak	foo
	.p2align	4
	.type	foo,@function
foo:
.Lfunc_begin0:
	li 3, 0
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	foo, .Lfunc_end0-.Lfunc_begin0

	.globl	main
	.p2align	4
	.type	main,@function
main:
.Lfunc_begin1:
.Lfunc_gep1:
	addis 2, 12, .TOC.-.Lfunc_gep1@ha
	addi 2, 2, .TOC.-.Lfunc_gep1@l
.Lfunc_lep1:
	.localentry	main, .Lfunc_lep1-.Lfunc_gep1
	mflr 0
	stdu 1, -32(1)
	std 0, 48(1)
	bl foo
	nop
	addi 1, 1, 32
	ld 0, 16(1)
	mtlr 0
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	main, .Lfunc_end1-.Lfunc_begin1
