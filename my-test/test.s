	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0"
	.file	"test.c"
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.text
	.globl	f                               # -- Begin function f
	.p2align	1
	.type	f,@function
f:                                      # @f
# %bb.0:
	addi	sp, sp, -32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	addi	s0, sp, 32
	sd	a0, -24(s0)
	ld	a0, -24(s0)
	lw	a0, 0(a0)
	sw	a0, -28(s0)
	lw	a0, -28(s0)
	addiw	a0, a0, 1
	sw	a0, -28(s0)
	lw	a0, -28(s0)
	ld	a1, -24(s0)
	sw	a0, 0(a1)
	lw	a0, -28(s0)
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	addi	sp, sp, 32
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
                                        # -- End function
	.option	pop
	.ident	"clang version 22.0.0git (https://github.com/llvm/llvm-project.git 34109cd26ae1b317d91c061500d9828fe6ebab0b)"
	.section	".note.GNU-stack","",@progbits
