	.attribute	4, 16
	.attribute	5, "rv32i2p1"
	.file	"shifts.ll"
	.text
	.globl	test_shl                        # -- Begin function test_shl
	.p2align	2
	.type	test_shl,@function
test_shl:                               # @test_shl
# %bb.0:                                # %entry
	addi	sp, sp, -16
                                        # kill: def $x13 killed $x12
                                        # kill: def $x13 killed $x11
                                        # kill: def $x13 killed $x10
	shift	a1, a1, a2
	not	a4, a2
	sri	a3, a0, 1
	shift	a3, a3, a4
	or	a1, a1, a3
	shift	a0, a0, a2
	sw	a0, 4(sp)                       # 4-byte Folded Spill
	addi	a0, a2, -32
	sw	a0, 8(sp)                       # 4-byte Folded Spill
	sw	a1, 12(sp)                      # 4-byte Folded Spill
	bltz	a0, .LBB0_2
# %bb.1:                                # %entry
	lw	a0, 4(sp)                       # 4-byte Folded Reload
	sw	a0, 12(sp)                      # 4-byte Folded Spill
.LBB0_2:                                # %entry
	lw	a0, 8(sp)                       # 4-byte Folded Reload
	lw	a2, 4(sp)                       # 4-byte Folded Reload
	lw	a1, 12(sp)                      # 4-byte Folded Reload
	sri	a0, a0, 31
	and	a0, a0, a2
	addi	sp, sp, 16
	ret
.Lfunc_end0:
	.size	test_shl, .Lfunc_end0-test_shl
                                        # -- End function
	.globl	test_lshr                       # -- Begin function test_lshr
	.p2align	2
	.type	test_lshr,@function
test_lshr:                              # @test_lshr
# %bb.0:                                # %entry
	addi	sp, sp, -16
	sw	a1, 0(sp)                       # 4-byte Folded Spill
	mv	a1, a0
	lw	a0, 0(sp)                       # 4-byte Folded Reload
                                        # kill: def $x13 killed $x12
                                        # kill: def $x13 killed $x10
                                        # kill: def $x13 killed $x11
	shift	a1, a1, a2
	not	a4, a2
	sli	a3, a0, 1
	shift	a3, a3, a4
	or	a1, a1, a3
	shift	a0, a0, a2
	sw	a0, 4(sp)                       # 4-byte Folded Spill
	addi	a0, a2, -32
	sw	a0, 8(sp)                       # 4-byte Folded Spill
	sw	a1, 12(sp)                      # 4-byte Folded Spill
	bltz	a0, .LBB1_2
# %bb.1:                                # %entry
	lw	a0, 4(sp)                       # 4-byte Folded Reload
	sw	a0, 12(sp)                      # 4-byte Folded Spill
.LBB1_2:                                # %entry
	lw	a1, 8(sp)                       # 4-byte Folded Reload
	lw	a2, 4(sp)                       # 4-byte Folded Reload
	lw	a0, 12(sp)                      # 4-byte Folded Reload
	sri	a1, a1, 31
	and	a1, a1, a2
	addi	sp, sp, 16
	ret
.Lfunc_end1:
	.size	test_lshr, .Lfunc_end1-test_lshr
                                        # -- End function
	.globl	test_ashr                       # -- Begin function test_ashr
	.p2align	2
	.type	test_ashr,@function
test_ashr:                              # @test_ashr
# %bb.0:                                # %entry
	addi	sp, sp, -32
	sw	a3, 8(sp)                       # 4-byte Folded Spill
	mv	a3, a2
	lw	a2, 8(sp)                       # 4-byte Folded Reload
	sw	a3, 12(sp)                      # 4-byte Folded Spill
	mv	a3, a1
	mv	a1, a0
	lw	a0, 12(sp)                      # 4-byte Folded Reload
                                        # kill: def $x12 killed $x10
                                        # kill: def $x12 killed $x13
                                        # kill: def $x12 killed $x11
	shift	a1, a1, a0
	not	a4, a0
	sli	a2, a3, 1
	shift	a2, a2, a4
	or	a2, a1, a2
	shift	a1, a3, a0
	sw	a1, 16(sp)                      # 4-byte Folded Spill
	addi	a0, a0, -32
	sri	a3, a3, 31
	sw	a3, 20(sp)                      # 4-byte Folded Spill
	sw	a2, 24(sp)                      # 4-byte Folded Spill
	sw	a1, 28(sp)                      # 4-byte Folded Spill
	bltz	a0, .LBB2_2
# %bb.1:                                # %entry
	lw	a0, 20(sp)                      # 4-byte Folded Reload
	lw	a1, 16(sp)                      # 4-byte Folded Reload
	sw	a1, 24(sp)                      # 4-byte Folded Spill
	sw	a0, 28(sp)                      # 4-byte Folded Spill
.LBB2_2:                                # %entry
	lw	a0, 24(sp)                      # 4-byte Folded Reload
	lw	a1, 28(sp)                      # 4-byte Folded Reload
	addi	sp, sp, 32
	ret
.Lfunc_end2:
	.size	test_ashr, .Lfunc_end2-test_ashr
                                        # -- End function
	.globl	run                             # -- Begin function run
	.p2align	2
	.type	run,@function
run:                                    # @run
# %bb.0:                                # %entry
	addi	sp, sp, -48
	sw	ra, 44(sp)                      # 4-byte Folded Spill
	sw	a3, 16(sp)                      # 4-byte Folded Spill
	sw	a2, 12(sp)                      # 4-byte Folded Spill
	sw	a1, 24(sp)                      # 4-byte Folded Spill
	sw	a0, 20(sp)                      # 4-byte Folded Spill
                                        # kill: def $x14 killed $x13
                                        # kill: def $x14 killed $x12
                                        # kill: def $x14 killed $x11
                                        # kill: def $x14 killed $x10
	call	test_shl
	lw	a2, 12(sp)                      # 4-byte Folded Reload
	lw	a3, 16(sp)                      # 4-byte Folded Reload
	mv	a4, a0
	lw	a0, 20(sp)                      # 4-byte Folded Reload
	sw	a4, 36(sp)                      # 4-byte Folded Spill
	mv	a4, a1
	lw	a1, 24(sp)                      # 4-byte Folded Reload
	sw	a4, 40(sp)                      # 4-byte Folded Spill
	call	test_lshr
	lw	a2, 12(sp)                      # 4-byte Folded Reload
	lw	a3, 16(sp)                      # 4-byte Folded Reload
	mv	a4, a0
	lw	a0, 20(sp)                      # 4-byte Folded Reload
	sw	a4, 32(sp)                      # 4-byte Folded Spill
	mv	a4, a1
	lw	a1, 24(sp)                      # 4-byte Folded Reload
	sw	a4, 28(sp)                      # 4-byte Folded Spill
	call	test_ashr
	lw	a5, 28(sp)                      # 4-byte Folded Reload
	lw	a4, 32(sp)                      # 4-byte Folded Reload
	mv	a3, a0
	lw	a0, 36(sp)                      # 4-byte Folded Reload
	mv	a2, a1
	lw	a1, 40(sp)                      # 4-byte Folded Reload
	xor	a1, a1, a5
	xor	a0, a0, a4
	xor	a0, a0, a3
	xor	a2, a1, a2
	lui	a1, %hi(sink)
	sw	a2, %lo(sink+4)(a1)
	sw	a0, %lo(sink)(a1)
	lw	ra, 44(sp)                      # 4-byte Folded Reload
	addi	sp, sp, 48
	ret
.Lfunc_end3:
	.size	run, .Lfunc_end3-run
                                        # -- End function
	.type	sink,@object                    # @sink
	.bss
	.globl	sink
	.p2align	3, 0x0
sink:
	.quad	0                               # 0x0
	.size	sink, 8

	.section	".note.GNU-stack","",@progbits
