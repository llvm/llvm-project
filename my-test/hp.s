	.file	"hp.ll"
	.text
	.globl	hp                              // -- Begin function hp
	.p2align	2
	.type	hp,@function
hp:                                     // @hp
	.cfi_startproc
// %bb.0:                               // %entry
	mov	w0, #36864                      // =0x9000
	movk	w0, #976, lsl #16
	ret
.Lfunc_end0:
	.size	hp, .Lfunc_end0-hp
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
