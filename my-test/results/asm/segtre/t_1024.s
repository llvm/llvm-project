	.file	"vreg-stress.ll"
	.text
	.globl	f                               // -- Begin function f
	.p2align	2
	.type	f,@function
f:                                      // @f
	.cfi_startproc
// %bb.0:                               // %entry
	mov	w1, w0
	mov	w0, #523776                     // =0x7fe00
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w1, w1, w1
	add	w0, w1, w0
	ret
.Lfunc_end0:
	.size	f, .Lfunc_end0-f
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
