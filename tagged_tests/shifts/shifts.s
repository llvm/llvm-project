	.attribute	4, 16
	.attribute	5, "rv32i2p1"
	.file	"shifts.ll"
	.text
	.globl	test_shl32                      # -- Begin function test_shl32
	.p2align	2
	.type	test_shl32,@function
test_shl32:                             # @test_shl32
# %bb.0:                                # %entry
	sl	a0, a0, a1
	ret
.Lfunc_end0:
	.size	test_shl32, .Lfunc_end0-test_shl32
                                        # -- End function
	.globl	test_lshr32                     # -- Begin function test_lshr32
	.p2align	2
	.type	test_lshr32,@function
test_lshr32:                            # @test_lshr32
# %bb.0:                                # %entry
	sr	a0, a0, a1
	ret
.Lfunc_end1:
	.size	test_lshr32, .Lfunc_end1-test_lshr32
                                        # -- End function
	.globl	test_ashr32                     # -- Begin function test_ashr32
	.p2align	2
	.type	test_ashr32,@function
test_ashr32:                            # @test_ashr32
# %bb.0:                                # %entry
	sr	a0, a0, a1
	ret
.Lfunc_end2:
	.size	test_ashr32, .Lfunc_end2-test_ashr32
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
