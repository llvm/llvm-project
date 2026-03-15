	.attribute	4, 16
	.attribute	5, "rv32i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0"
	.file	"sltTest.c"
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.text
	.globl	f1                              # -- Begin function f1
	.p2align	1
	.type	f1,@function
f1:                                     # @f1
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	slt	a0, a0, a1
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end0:
	.size	f1, .Lfunc_end0-f1
                                        # -- End function
	.option	pop
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.globl	f2                              # -- Begin function f2
	.p2align	1
	.type	f2,@function
f2:                                     # @f2
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	slt	a0, a0, a1
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end1:
	.size	f2, .Lfunc_end1-f2
                                        # -- End function
	.option	pop
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.globl	f3                              # -- Begin function f3
	.p2align	1
	.type	f3,@function
f3:                                     # @f3
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	slti	a0, a0, 4
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end2:
	.size	f3, .Lfunc_end2-f3
                                        # -- End function
	.option	pop
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.globl	f4                              # -- Begin function f4
	.p2align	1
	.type	f4,@function
f4:                                     # @f4
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	slti	a0, a0, 5
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end3:
	.size	f4, .Lfunc_end3-f4
                                        # -- End function
	.option	pop
	.ident	"Homebrew clang version 22.1.0"
	.section	".note.GNU-stack","",@progbits
