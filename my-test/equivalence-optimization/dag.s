	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0"
	.file	"foo.c"
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.text
	.globl	foo                             # -- Begin function foo
	.p2align	1
	.type	foo,@function
foo:                                    # @foo
# %bb.0:
	addi	sp, sp, -32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	addi	s0, sp, 32
	sw	a0, -24(s0)
	sw	a1, -28(s0)
	lw	a0, -24(s0)
	lw	a1, -28(s0)
	and	a0, a0, a1
	bne	a0, a1, .LBB0_2
	j	.LBB0_1
.LBB0_1:
	lw	a0, -24(s0)
	lw	a1, -28(s0)
	subw	a0, a0, a1
	sw	a0, -20(s0)
	j	.LBB0_3
.LBB0_2:
	sw	zero, -20(s0)
	j	.LBB0_3
.LBB0_3:
	lw	a0, -20(s0)
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	addi	sp, sp, 32
	ret
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
                                        # -- End function
	.option	pop
	.ident	"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 34109cd26ae1b317d91c061500d9828fe6ebab0b)"
	.section	".note.GNU-stack","",@progbits
