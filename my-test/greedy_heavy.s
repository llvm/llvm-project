	.file	"test_heavy_pressure.ll"
	.text
	.globl	heavy_pressure                  // -- Begin function heavy_pressure
	.p2align	2
	.type	heavy_pressure,@function
heavy_pressure:                         // @heavy_pressure
	.cfi_startproc
// %bb.0:                               // %entry
	str	x19, [sp, #-16]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset w19, -16
	mov	w8, #11                         // =0xb
	mov	w9, #13                         // =0xd
	lsl	w10, w0, #1
	add	w11, w0, w0, lsl #2
	mov	w12, #19                        // =0x13
	mov	w16, #29                        // =0x1d
	mul	w8, w0, w8
	lsl	w13, w0, #3
	mov	w14, #23                        // =0x17
	mul	w9, w0, w9
	mov	w17, #37                        // =0x25
	lsl	w2, w0, #5
	mul	w12, w0, w12
	add	w15, w10, w0
	add	w18, w0, w0, lsl #4
	mul	w16, w0, w16
	mov	w1, #41                         // =0x29
	mov	w3, #43                         // =0x2b
	mul	w14, w0, w14
	mov	w4, #47                         // =0x2f
	add	w5, w10, w11
	mul	w17, w0, w17
	sub	w13, w13, w0
	mov	w6, #53                         // =0x35
	mul	w1, w0, w1
	add	w5, w5, w15
	sub	w2, w2, w0
	mul	w3, w0, w3
	add	w5, w5, w13
	add	w7, w18, w12
	mul	w4, w0, w4
	add	w19, w16, w2
	mul	w0, w0, w6
	add	w6, w8, w9
	add	w5, w5, w6
	add	w6, w7, w14
	add	w7, w19, w17
	add	w5, w5, w6
	add	w6, w7, w1
	add	w7, w3, w4
	add	w5, w5, w6
	add	w6, w7, w0
	add	w5, w5, w6
	madd	w10, w10, w0, w5
	madd	w10, w15, w4, w10
	madd	w10, w11, w3, w10
	madd	w10, w13, w1, w10
	madd	w8, w8, w17, w10
	madd	w8, w9, w2, w8
	madd	w8, w18, w16, w8
	madd	w0, w12, w14, w8
	ldr	x19, [sp], #16                  // 8-byte Folded Reload
	ret
.Lfunc_end0:
	.size	heavy_pressure, .Lfunc_end0-heavy_pressure
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
