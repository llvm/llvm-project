	.file	"arm64-int-neon.ll"
	.text
	.globl	test_sqrshl_s32                 // -- Begin function test_sqrshl_s32
	.p2align	2
	.type	test_sqrshl_s32,@function
test_sqrshl_s32:                        // @test_sqrshl_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	sqrshl	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end0:
	.size	test_sqrshl_s32, .Lfunc_end0-test_sqrshl_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqrshl_s64                 // -- Begin function test_sqrshl_s64
	.p2align	2
	.type	test_sqrshl_s64,@function
test_sqrshl_s64:                        // @test_sqrshl_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqrshl	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end1:
	.size	test_sqrshl_s64, .Lfunc_end1-test_sqrshl_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_sqshl_s32                  // -- Begin function test_sqshl_s32
	.p2align	2
	.type	test_sqshl_s32,@function
test_sqshl_s32:                         // @test_sqshl_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	sqshl	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end2:
	.size	test_sqshl_s32, .Lfunc_end2-test_sqshl_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqshl_s64                  // -- Begin function test_sqshl_s64
	.p2align	2
	.type	test_sqshl_s64,@function
test_sqshl_s64:                         // @test_sqshl_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqshl	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end3:
	.size	test_sqshl_s64, .Lfunc_end3-test_sqshl_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_uqrshl_s32                 // -- Begin function test_uqrshl_s32
	.p2align	2
	.type	test_uqrshl_s32,@function
test_uqrshl_s32:                        // @test_uqrshl_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	uqrshl	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end4:
	.size	test_uqrshl_s32, .Lfunc_end4-test_uqrshl_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqrshl_s64                 // -- Begin function test_uqrshl_s64
	.p2align	2
	.type	test_uqrshl_s64,@function
test_uqrshl_s64:                        // @test_uqrshl_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqrshl	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end5:
	.size	test_uqrshl_s64, .Lfunc_end5-test_uqrshl_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_uqshl_s32                  // -- Begin function test_uqshl_s32
	.p2align	2
	.type	test_uqshl_s32,@function
test_uqshl_s32:                         // @test_uqshl_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	uqshl	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end6:
	.size	test_uqshl_s32, .Lfunc_end6-test_uqshl_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqshl_s64                  // -- Begin function test_uqshl_s64
	.p2align	2
	.type	test_uqshl_s64,@function
test_uqshl_s64:                         // @test_uqshl_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqshl	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end7:
	.size	test_uqshl_s64, .Lfunc_end7-test_uqshl_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_sqshrn_s32                 // -- Begin function test_sqshrn_s32
	.p2align	2
	.type	test_sqshrn_s32,@function
test_sqshrn_s32:                        // @test_sqshrn_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqshrn	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end8:
	.size	test_sqshrn_s32, .Lfunc_end8-test_sqshrn_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqshrun_s32                // -- Begin function test_sqshrun_s32
	.p2align	2
	.type	test_sqshrun_s32,@function
test_sqshrun_s32:                       // @test_sqshrun_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqshrun	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end9:
	.size	test_sqshrun_s32, .Lfunc_end9-test_sqshrun_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqshrn_s32                 // -- Begin function test_uqshrn_s32
	.p2align	2
	.type	test_uqshrn_s32,@function
test_uqshrn_s32:                        // @test_uqshrn_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqshrn	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end10:
	.size	test_uqshrn_s32, .Lfunc_end10-test_uqshrn_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqrshrn_s32                // -- Begin function test_sqrshrn_s32
	.p2align	2
	.type	test_sqrshrn_s32,@function
test_sqrshrn_s32:                       // @test_sqrshrn_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqrshrn	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end11:
	.size	test_sqrshrn_s32, .Lfunc_end11-test_sqrshrn_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqrshrun_s32               // -- Begin function test_sqrshrun_s32
	.p2align	2
	.type	test_sqrshrun_s32,@function
test_sqrshrun_s32:                      // @test_sqrshrun_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqrshrun	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end12:
	.size	test_sqrshrun_s32, .Lfunc_end12-test_sqrshrun_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqrshrn_s32                // -- Begin function test_uqrshrn_s32
	.p2align	2
	.type	test_uqrshrn_s32,@function
test_uqrshrn_s32:                       // @test_uqrshrn_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqrshrn	s0, d0, #1
	str	s0, [x0]
	ret
.Lfunc_end13:
	.size	test_uqrshrn_s32, .Lfunc_end13-test_uqrshrn_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqadd_s32                  // -- Begin function test_sqadd_s32
	.p2align	2
	.type	test_sqadd_s32,@function
test_sqadd_s32:                         // @test_sqadd_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	sqadd	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end14:
	.size	test_sqadd_s32, .Lfunc_end14-test_sqadd_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqadd_s64                  // -- Begin function test_sqadd_s64
	.p2align	2
	.type	test_sqadd_s64,@function
test_sqadd_s64:                         // @test_sqadd_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqadd	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end15:
	.size	test_sqadd_s64, .Lfunc_end15-test_sqadd_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_sqsub_s32                  // -- Begin function test_sqsub_s32
	.p2align	2
	.type	test_sqsub_s32,@function
test_sqsub_s32:                         // @test_sqsub_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	sqsub	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end16:
	.size	test_sqsub_s32, .Lfunc_end16-test_sqsub_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_sqsub_s64                  // -- Begin function test_sqsub_s64
	.p2align	2
	.type	test_sqsub_s64,@function
test_sqsub_s64:                         // @test_sqsub_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	sqsub	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end17:
	.size	test_sqsub_s64, .Lfunc_end17-test_sqsub_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_uqadd_s32                  // -- Begin function test_uqadd_s32
	.p2align	2
	.type	test_uqadd_s32,@function
test_uqadd_s32:                         // @test_uqadd_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	uqadd	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end18:
	.size	test_uqadd_s32, .Lfunc_end18-test_uqadd_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqadd_s64                  // -- Begin function test_uqadd_s64
	.p2align	2
	.type	test_uqadd_s64,@function
test_uqadd_s64:                         // @test_uqadd_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqadd	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end19:
	.size	test_uqadd_s64, .Lfunc_end19-test_uqadd_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_uqsub_s32                  // -- Begin function test_uqsub_s32
	.p2align	2
	.type	test_uqsub_s32,@function
test_uqsub_s32:                         // @test_uqsub_s32
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	s0, s0
	uqsub	s0, s0, s0
	fmov	w0, s0
	ret
.Lfunc_end20:
	.size	test_uqsub_s32, .Lfunc_end20-test_uqsub_s32
	.cfi_endproc
                                        // -- End function
	.globl	test_uqsub_s64                  // -- Begin function test_uqsub_s64
	.p2align	2
	.type	test_uqsub_s64,@function
test_uqsub_s64:                         // @test_uqsub_s64
	.cfi_startproc
// %bb.0:                               // %entry
	fcvtzs	d0, s0
	uqsub	d0, d0, d0
	fmov	x0, d0
	ret
.Lfunc_end21:
	.size	test_uqsub_s64, .Lfunc_end21-test_uqsub_s64
	.cfi_endproc
                                        // -- End function
	.globl	test_sqdmulls_scalar            // -- Begin function test_sqdmulls_scalar
	.p2align	2
	.type	test_sqdmulls_scalar,@function
test_sqdmulls_scalar:                   // @test_sqdmulls_scalar
	.cfi_startproc
// %bb.0:
	fcvtzs	s0, s0
	sqdmull	d0, s0, s0
	fmov	x0, d0
	ret
.Lfunc_end22:
	.size	test_sqdmulls_scalar, .Lfunc_end22-test_sqdmulls_scalar
	.cfi_endproc
                                        // -- End function
	.section	".note.GNU-stack","",@progbits
