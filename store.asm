	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_b1p0_zmmul1p0_zba1p0_zbb1p0_zbs1p0"
	.file	"store.ll"
	.text
	.globl	store_i64                       # -- Begin function store_i64
	.p2align	2
	.type	store_i64,@function
store_i64:                              # @store_i64
	.cfi_startproc
# %bb.0:
	sd	a1, 0(a0)
	ret
.Lfunc_end0:
	.size	store_i64, .Lfunc_end0-store_i64
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
