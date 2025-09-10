	.file	"vreg-stress.ll"
	.text
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:                               // %entry
	add	w9, w0, w0
	mov	w8, #523776                     // =0x7fe00
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w9, w9, w9
	add	w0, w9, w8
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
