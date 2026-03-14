	.text
	.abiversion 2

	.globl	lfunc
	.p2align	4
	.type	lfunc,@function
lfunc:                                  # @lfunc
.Lfunc_begin0:
.Lfunc_gep0:
	addis 2, 12, .TOC.-.Lfunc_gep0@ha
	addi 2, 2, .TOC.-.Lfunc_gep0@l
.Lfunc_lep0:
	.localentry	lfunc, .Lfunc_lep0-.Lfunc_gep0
# BB#0:
	mr 4, 3
	addis 3, 2, .LC0@toc@ha
	ld 3, .LC0@toc@l(3)
	stw 4, -12(1)
	lwz 4, 0(3)
	lwz 5, -12(1)
	mullw 4, 4, 5
	extsw 3, 4
	blr
	.long	0
	.quad	0
.Lfunc_end0:
	.size	lfunc, .Lfunc_end0-.Lfunc_begin0

	.globl	simple
	.p2align	4
	.type	simple,@function
simple:                                 # @simple
.Lfunc_begin1:
# %bb.0:                                # %entry
	mr 4, 3
	stw 4, -12(1)
	lwz 4, -12(1)
	mulli 4, 4, 10
	extsw 3, 4
	blr
	.long	0
	.quad	0
.Lfunc_end1:
	.size	simple, .Lfunc_end1-.Lfunc_begin1

	.section	.toc,"aw",@progbits
.LC0:
	.tc g_foo[TC],g_foo
	.type	g_foo,@object           # @g_foo
	.data
	.globl	g_foo
	.p2align	2
g_foo:
	.long	2                       # 0x2
	.size	g_foo, 4
