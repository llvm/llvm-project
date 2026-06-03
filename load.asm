	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_b1p0_zmmul1p0_zba1p0_zbb1p0_zbs1p0"
	.file	"load.ll"
	.text
	.globl	load_i64                        # -- Begin function load_i64
	.p2align	2
	.type	load_i64,@function
load_i64:                               # @load_i64
	.cfi_startproc
# %bb.0:
	ld	a0, 0(a0)
	ret
.Lfunc_end0:
	.size	load_i64, .Lfunc_end0-load_i64
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
