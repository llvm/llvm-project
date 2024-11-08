	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_f2p2_d2p2_v1p0_zicsr2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"skip-mir-comment-trailing-whitespace.mir"
	.globl	test_vse8                       # -- Begin function test_vse8
	.type	test_vse8,@function
test_vse8:                              # @test_vse8
	.cfi_startproc
# %bb.0:
	vsetvli	zero, a1, e8, mf8, ta, ma
	vse8.v	v8, (a0)
	ret
.Lfunc_end0:
	.size	test_vse8, .Lfunc_end0-test_vse8
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
