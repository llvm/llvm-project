	.attribute	4, 16
	.attribute	5, "rv32i2p1_m2p0_a2p1_c2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0"
	.file	"immShifts.c"
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.text
	.globl	shl                             # -- Begin function shl
	.p2align	1
	.type	shl,@function
shl:                                    # @shl
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	sli	a0, a0, 3
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end0:
	.size	shl, .Lfunc_end0-shl
                                        # -- End function
	.option	pop
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.globl	asr                             # -- Begin function asr
	.p2align	1
	.type	asr,@function
asr:                                    # @asr
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	sri	a0, a0, 3
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end1:
	.size	asr, .Lfunc_end1-asr
                                        # -- End function
	.option	pop
	.option	push
	.option	arch, +a, +c, +m, +zaamo, +zalrsc, +zca, +zmmul
	.globl	lsr                             # -- Begin function lsr
	.p2align	1
	.type	lsr,@function
lsr:                                    # @lsr
# %bb.0:
	addi	sp, sp, -16
	sw	ra, 12(sp)                      # 4-byte Folded Spill
	sw	s0, 8(sp)                       # 4-byte Folded Spill
	addi	s0, sp, 16
	sri	a0, a0, 3
	lw	ra, 12(sp)                      # 4-byte Folded Reload
	lw	s0, 8(sp)                       # 4-byte Folded Reload
	addi	sp, sp, 16
	ret
.Lfunc_end2:
	.size	lsr, .Lfunc_end2-lsr
                                        # -- End function
	.option	pop
	.ident	"clang version 23.0.0git (https://github.com/llvm/llvm-project.git b397c9d24196fe4103ad7cfe6bbfdfc08496364b)"
	.section	".note.GNU-stack","",@progbits
