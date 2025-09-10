	.file	"test_spill.ll"
	.text
	.globl	high_pressure                   // -- Begin function high_pressure
	.p2align	2
	.type	high_pressure,@function
high_pressure:                          // @high_pressure
	.cfi_startproc
// %bb.0:                               // %entry
	mov	w0, #815                        // =0x32f
	ret
.Lfunc_end0:
	.size	high_pressure, .Lfunc_end0-high_pressure
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
