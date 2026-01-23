	.text
	.attribute	4, 16
	.attribute	5, "rv32i2p1_m2p0_a2p1_c2p0"
	.file	"test_lrsc_rv32.c"
	.globl	cas_set_42                      # -- Begin function cas_set_42
	.p2align	1
	.type	cas_set_42,@function
cas_set_42:                             # @cas_set_42
# %bb.0:
	lui	a0, %hi(g)
	addi	a0, a0, %lo(g)
	li	a1, 42
.LBB0_1:                                # =>This Inner Loop Header: Depth=1
	lr.w.aqrl	a2, (a0)
	bnez	a2, .LBB0_3
# %bb.2:                                #   in Loop: Header=BB0_1 Depth=1
	sc.w.rl	a3, a1, (a0)
	bnez	a3, .LBB0_1
.LBB0_3:
	seqz	a0, a2
	ret
.Lfunc_end0:
	.size	cas_set_42, .Lfunc_end0-cas_set_42
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   # @main
# %bb.0:
	lui	a0, %hi(g)
	addi	a0, a0, %lo(g)
	li	a1, 42
.LBB1_1:                                # =>This Inner Loop Header: Depth=1
	lr.w.aqrl	a2, (a0)
	bnez	a2, .LBB1_3
# %bb.2:                                #   in Loop: Header=BB1_1 Depth=1
	sc.w.rl	a3, a1, (a0)
	bnez	a3, .LBB1_1
.LBB1_3:
	snez	a0, a2
	ret
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
                                        # -- End function
	.type	g,@object                       # @g
	.section	.sbss,"aw",@nobits
	.globl	g
	.p2align	2, 0x0
g:
	.word	0                               # 0x0
	.size	g, 4

	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
