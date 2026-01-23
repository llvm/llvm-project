	.attribute	4, 16
	.attribute	5, "rv64i2p1"
	.file	"one_lrsc.c"
	.option	push
	.option	arch, +a, +zaamo, +zalrsc
	.text
	.globl	cas_u32                         # -- Begin function cas_u32
	.p2align	2
	.type	cas_u32,@function
cas_u32:                                # @cas_u32
# %bb.0:
	lui	a2, %hi(g)
	addi	a2, a2, %lo(g)
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	lr.w.aqrl	a3, (a2)
	bne	a3, a0, .LBB0_3
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	sc.w.rl	a4, a1, (a2)
	bnez	a4, .LBB0_1
.LBB0_3:
	xor	a0, a3, a0
	seqz	a0, a0
	ret
.Lfunc_end0:
	.size	cas_u32, .Lfunc_end0-cas_u32
                                        # -- End function
	.option	pop
	.type	g,@object                       # @g
	.section	.sbss,"aw",@nobits
	.globl	g
	.p2align	2, 0x0
g:
	.word	0                               # 0x0
	.size	g, 4

	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
