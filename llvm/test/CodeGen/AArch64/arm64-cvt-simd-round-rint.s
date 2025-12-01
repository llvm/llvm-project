	.file	"arm64-cvt-simd-round-rint.ll"
	.text
	.globl	lround_i32_f16_simd             // -- Begin function lround_i32_f16_simd
	.p2align	2
	.type	lround_i32_f16_simd,@function
lround_i32_f16_simd:                    // @lround_i32_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtas	x8, h0
	fmov	s0, w8
	ret
.Lfunc_end0:
	.size	lround_i32_f16_simd, .Lfunc_end0-lround_i32_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f16_simd             // -- Begin function lround_i64_f16_simd
	.p2align	2
	.type	lround_i64_f16_simd,@function
lround_i64_f16_simd:                    // @lround_i64_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, h0
	ret
.Lfunc_end1:
	.size	lround_i64_f16_simd, .Lfunc_end1-lround_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f32_simd             // -- Begin function lround_i64_f32_simd
	.p2align	2
	.type	lround_i64_f32_simd,@function
lround_i64_f32_simd:                    // @lround_i64_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end2:
	.size	lround_i64_f32_simd, .Lfunc_end2-lround_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i32_f64_simd             // -- Begin function lround_i32_f64_simd
	.p2align	2
	.type	lround_i32_f64_simd,@function
lround_i32_f64_simd:                    // @lround_i32_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtas	x8, d0
	fmov	s0, w8
	ret
.Lfunc_end3:
	.size	lround_i32_f64_simd, .Lfunc_end3-lround_i32_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i32_f32_simd             // -- Begin function lround_i32_f32_simd
	.p2align	2
	.type	lround_i32_f32_simd,@function
lround_i32_f32_simd:                    // @lround_i32_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtas	x8, s0
	fmov	s0, w8
	ret
.Lfunc_end4:
	.size	lround_i32_f32_simd, .Lfunc_end4-lround_i32_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f64_simd             // -- Begin function lround_i64_f64_simd
	.p2align	2
	.type	lround_i64_f64_simd,@function
lround_i64_f64_simd:                    // @lround_i64_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end5:
	.size	lround_i64_f64_simd, .Lfunc_end5-lround_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f16_simd            // -- Begin function llround_i64_f16_simd
	.p2align	2
	.type	llround_i64_f16_simd,@function
llround_i64_f16_simd:                   // @llround_i64_f16_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, h0
	ret
.Lfunc_end6:
	.size	llround_i64_f16_simd, .Lfunc_end6-llround_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f32_simd            // -- Begin function llround_i64_f32_simd
	.p2align	2
	.type	llround_i64_f32_simd,@function
llround_i64_f32_simd:                   // @llround_i64_f32_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end7:
	.size	llround_i64_f32_simd, .Lfunc_end7-llround_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f64_simd            // -- Begin function llround_i64_f64_simd
	.p2align	2
	.type	llround_i64_f64_simd,@function
llround_i64_f64_simd:                   // @llround_i64_f64_simd
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end8:
	.size	llround_i64_f64_simd, .Lfunc_end8-llround_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	lround_i32_f16_simd_exp         // -- Begin function lround_i32_f16_simd_exp
	.p2align	2
	.type	lround_i32_f16_simd_exp,@function
lround_i32_f16_simd_exp:                // @lround_i32_f16_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	s0, h0
	ret
.Lfunc_end9:
	.size	lround_i32_f16_simd_exp, .Lfunc_end9-lround_i32_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f16_simd_exp         // -- Begin function lround_i64_f16_simd_exp
	.p2align	2
	.type	lround_i64_f16_simd_exp,@function
lround_i64_f16_simd_exp:                // @lround_i64_f16_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, h0
	ret
.Lfunc_end10:
	.size	lround_i64_f16_simd_exp, .Lfunc_end10-lround_i64_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f32_simd_exp         // -- Begin function lround_i64_f32_simd_exp
	.p2align	2
	.type	lround_i64_f32_simd_exp,@function
lround_i64_f32_simd_exp:                // @lround_i64_f32_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end11:
	.size	lround_i64_f32_simd_exp, .Lfunc_end11-lround_i64_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lround_i32_f64_simd_exp         // -- Begin function lround_i32_f64_simd_exp
	.p2align	2
	.type	lround_i32_f64_simd_exp,@function
lround_i32_f64_simd_exp:                // @lround_i32_f64_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	s0, d0
	ret
.Lfunc_end12:
	.size	lround_i32_f64_simd_exp, .Lfunc_end12-lround_i32_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lround_i32_f32_simd_exp         // -- Begin function lround_i32_f32_simd_exp
	.p2align	2
	.type	lround_i32_f32_simd_exp,@function
lround_i32_f32_simd_exp:                // @lround_i32_f32_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	s0, s0
	ret
.Lfunc_end13:
	.size	lround_i32_f32_simd_exp, .Lfunc_end13-lround_i32_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lround_i64_f64_simd_exp         // -- Begin function lround_i64_f64_simd_exp
	.p2align	2
	.type	lround_i64_f64_simd_exp,@function
lround_i64_f64_simd_exp:                // @lround_i64_f64_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end14:
	.size	lround_i64_f64_simd_exp, .Lfunc_end14-lround_i64_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f16_simd_exp        // -- Begin function llround_i64_f16_simd_exp
	.p2align	2
	.type	llround_i64_f16_simd_exp,@function
llround_i64_f16_simd_exp:               // @llround_i64_f16_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, h0
	ret
.Lfunc_end15:
	.size	llround_i64_f16_simd_exp, .Lfunc_end15-llround_i64_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f32_simd_exp        // -- Begin function llround_i64_f32_simd_exp
	.p2align	2
	.type	llround_i64_f32_simd_exp,@function
llround_i64_f32_simd_exp:               // @llround_i64_f32_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, s0
	ret
.Lfunc_end16:
	.size	llround_i64_f32_simd_exp, .Lfunc_end16-llround_i64_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llround_i64_f64_simd_exp        // -- Begin function llround_i64_f64_simd_exp
	.p2align	2
	.type	llround_i64_f64_simd_exp,@function
llround_i64_f64_simd_exp:               // @llround_i64_f64_simd_exp
	.cfi_startproc
// %bb.0:
	fcvtas	d0, d0
	ret
.Lfunc_end17:
	.size	llround_i64_f64_simd_exp, .Lfunc_end17-llround_i64_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f16_simd              // -- Begin function lrint_i32_f16_simd
	.p2align	2
	.type	lrint_i32_f16_simd,@function
lrint_i32_f16_simd:                     // @lrint_i32_f16_simd
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	s0, h0
	ret
.Lfunc_end18:
	.size	lrint_i32_f16_simd, .Lfunc_end18-lrint_i32_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f16_simd              // -- Begin function lrint_i64_f16_simd
	.p2align	2
	.type	lrint_i64_f16_simd,@function
lrint_i64_f16_simd:                     // @lrint_i64_f16_simd
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	d0, h0
	ret
.Lfunc_end19:
	.size	lrint_i64_f16_simd, .Lfunc_end19-lrint_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f32_simd              // -- Begin function lrint_i64_f32_simd
	.p2align	2
	.type	lrint_i64_f32_simd,@function
lrint_i64_f32_simd:                     // @lrint_i64_f32_simd
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	d0, s0
	ret
.Lfunc_end20:
	.size	lrint_i64_f32_simd, .Lfunc_end20-lrint_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f64_simd              // -- Begin function lrint_i32_f64_simd
	.p2align	2
	.type	lrint_i32_f64_simd,@function
lrint_i32_f64_simd:                     // @lrint_i32_f64_simd
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	s0, d0
	ret
.Lfunc_end21:
	.size	lrint_i32_f64_simd, .Lfunc_end21-lrint_i32_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f32_simd              // -- Begin function lrint_i32_f32_simd
	.p2align	2
	.type	lrint_i32_f32_simd,@function
lrint_i32_f32_simd:                     // @lrint_i32_f32_simd
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	s0, s0
	ret
.Lfunc_end22:
	.size	lrint_i32_f32_simd, .Lfunc_end22-lrint_i32_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f64_simd              // -- Begin function lrint_i64_f64_simd
	.p2align	2
	.type	lrint_i64_f64_simd,@function
lrint_i64_f64_simd:                     // @lrint_i64_f64_simd
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	d0, d0
	ret
.Lfunc_end23:
	.size	lrint_i64_f64_simd, .Lfunc_end23-lrint_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f16_simd             // -- Begin function llrint_i64_f16_simd
	.p2align	2
	.type	llrint_i64_f16_simd,@function
llrint_i64_f16_simd:                    // @llrint_i64_f16_simd
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	d0, h0
	ret
.Lfunc_end24:
	.size	llrint_i64_f16_simd, .Lfunc_end24-llrint_i64_f16_simd
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f32_simd             // -- Begin function llrint_i64_f32_simd
	.p2align	2
	.type	llrint_i64_f32_simd,@function
llrint_i64_f32_simd:                    // @llrint_i64_f32_simd
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	d0, s0
	ret
.Lfunc_end25:
	.size	llrint_i64_f32_simd, .Lfunc_end25-llrint_i64_f32_simd
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f64_simd             // -- Begin function llrint_i64_f64_simd
	.p2align	2
	.type	llrint_i64_f64_simd,@function
llrint_i64_f64_simd:                    // @llrint_i64_f64_simd
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	d0, d0
	ret
.Lfunc_end26:
	.size	llrint_i64_f64_simd, .Lfunc_end26-llrint_i64_f64_simd
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f16_simd_exp          // -- Begin function lrint_i32_f16_simd_exp
	.p2align	2
	.type	lrint_i32_f16_simd_exp,@function
lrint_i32_f16_simd_exp:                 // @lrint_i32_f16_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	s0, h0
	ret
.Lfunc_end27:
	.size	lrint_i32_f16_simd_exp, .Lfunc_end27-lrint_i32_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f16_simd_exp          // -- Begin function lrint_i64_f16_simd_exp
	.p2align	2
	.type	lrint_i64_f16_simd_exp,@function
lrint_i64_f16_simd_exp:                 // @lrint_i64_f16_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	d0, h0
	ret
.Lfunc_end28:
	.size	lrint_i64_f16_simd_exp, .Lfunc_end28-lrint_i64_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f32_simd_exp          // -- Begin function lrint_i64_f32_simd_exp
	.p2align	2
	.type	lrint_i64_f32_simd_exp,@function
lrint_i64_f32_simd_exp:                 // @lrint_i64_f32_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	d0, s0
	ret
.Lfunc_end29:
	.size	lrint_i64_f32_simd_exp, .Lfunc_end29-lrint_i64_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f64_simd_exp          // -- Begin function lrint_i32_f64_simd_exp
	.p2align	2
	.type	lrint_i32_f64_simd_exp,@function
lrint_i32_f64_simd_exp:                 // @lrint_i32_f64_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	s0, d0
	ret
.Lfunc_end30:
	.size	lrint_i32_f64_simd_exp, .Lfunc_end30-lrint_i32_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i32_f32_simd_exp          // -- Begin function lrint_i32_f32_simd_exp
	.p2align	2
	.type	lrint_i32_f32_simd_exp,@function
lrint_i32_f32_simd_exp:                 // @lrint_i32_f32_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	s0, s0
	ret
.Lfunc_end31:
	.size	lrint_i32_f32_simd_exp, .Lfunc_end31-lrint_i32_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	lrint_i64_f64_simd_exp          // -- Begin function lrint_i64_f64_simd_exp
	.p2align	2
	.type	lrint_i64_f64_simd_exp,@function
lrint_i64_f64_simd_exp:                 // @lrint_i64_f64_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	d0, d0
	ret
.Lfunc_end32:
	.size	lrint_i64_f64_simd_exp, .Lfunc_end32-lrint_i64_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f16_simd_exp         // -- Begin function llrint_i64_f16_simd_exp
	.p2align	2
	.type	llrint_i64_f16_simd_exp,@function
llrint_i64_f16_simd_exp:                // @llrint_i64_f16_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	h0, h0
	fcvtzs	d0, h0
	ret
.Lfunc_end33:
	.size	llrint_i64_f16_simd_exp, .Lfunc_end33-llrint_i64_f16_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f32_simd_exp         // -- Begin function llrint_i64_f32_simd_exp
	.p2align	2
	.type	llrint_i64_f32_simd_exp,@function
llrint_i64_f32_simd_exp:                // @llrint_i64_f32_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	s0, s0
	fcvtzs	d0, s0
	ret
.Lfunc_end34:
	.size	llrint_i64_f32_simd_exp, .Lfunc_end34-llrint_i64_f32_simd_exp
	.cfi_endproc
                                        // -- End function
	.globl	llrint_i64_f64_simd_exp         // -- Begin function llrint_i64_f64_simd_exp
	.p2align	2
	.type	llrint_i64_f64_simd_exp,@function
llrint_i64_f64_simd_exp:                // @llrint_i64_f64_simd_exp
	.cfi_startproc
// %bb.0:
	frintx	d0, d0
	fcvtzs	d0, d0
	ret
.Lfunc_end35:
	.size	llrint_i64_f64_simd_exp, .Lfunc_end35-llrint_i64_f64_simd_exp
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
