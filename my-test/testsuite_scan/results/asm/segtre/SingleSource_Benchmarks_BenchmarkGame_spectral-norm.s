	.file	"spectral-norm.c"
	.text
	.globl	eval_A                          // -- Begin function eval_A
	.p2align	2
	.type	eval_A,@function
eval_A:                                 // @eval_A
	.cfi_startproc
// %bb.0:
	add	w1, w1, w0
	fmov	d1, #1.00000000
	madd	w1, w1, w1, w1
	add	w1, w1, w1, lsr #31
	add	w0, w0, w1, asr #1
	add	w0, w0, #1
	scvtf	d0, w0
	fdiv	d0, d1, d0
	ret
.Lfunc_end0:
	.size	eval_A, .Lfunc_end0-eval_A
	.cfi_endproc
                                        // -- End function
	.globl	eval_A_times_u                  // -- Begin function eval_A_times_u
	.p2align	2
	.type	eval_A_times_u,@function
eval_A_times_u:                         // @eval_A_times_u
	.cfi_startproc
// %bb.0:
	mov	x0, x2
	cmp	w0, #1
	b.lt	.LBB1_5
// %bb.1:
	fmov	d0, #1.00000000
	mov	x2, xzr
	mov	w3, w0
	mov	w4, #1                          // =0x1
.LBB1_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_3 Depth 2
	movi	d1, #0000000000000000
	add	x5, x2, #1
	mov	x6, x4
	mov	x7, x1
	mov	x8, x3
	str	xzr, [x0, x2, lsl #3]
.LBB1_3:                                //   Parent Loop BB1_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	w9, w6, #1
	ldr	d3, [x7], #8
	mul	w9, w6, w9
	subs	x8, x8, #1
	add	x6, x6, #1
	add	w9, w5, w9, lsr #1
	scvtf	d2, w9
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	str	d1, [x0, x2, lsl #3]
	b.ne	.LBB1_3
// %bb.4:                               //   in Loop: Header=BB1_2 Depth=1
	cmp	x5, x3
	add	x4, x4, #1
	mov	x2, x5
	b.ne	.LBB1_2
.LBB1_5:
	ret
.Lfunc_end1:
	.size	eval_A_times_u, .Lfunc_end1-eval_A_times_u
	.cfi_endproc
                                        // -- End function
	.globl	eval_At_times_u                 // -- Begin function eval_At_times_u
	.p2align	2
	.type	eval_At_times_u,@function
eval_At_times_u:                        // @eval_At_times_u
	.cfi_startproc
// %bb.0:
	mov	x0, x2
	cmp	w0, #1
	b.lt	.LBB2_5
// %bb.1:
	fmov	d0, #1.00000000
	mov	w2, wzr
	mov	x3, xzr
	mov	w4, w0
	mov	w5, #2                          // =0x2
.LBB2_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_3 Depth 2
	movi	d1, #0000000000000000
	mov	w6, #1                          // =0x1
	mov	w7, w2
	mov	w9, w5
	mov	x8, x4
	mov	x10, x1
	str	xzr, [x0, x3, lsl #3]
.LBB2_3:                                //   Parent Loop BB2_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	w11, w6, w7, lsr #1
	ldr	d3, [x10], #8
	subs	x8, x8, #1
	add	w7, w7, w9
	add	w9, w9, #2
	scvtf	d2, w11
	add	w6, w6, #1
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	str	d1, [x0, x3, lsl #3]
	b.ne	.LBB2_3
// %bb.4:                               //   in Loop: Header=BB2_2 Depth=1
	add	x3, x3, #1
	add	w6, w5, #2
	add	w2, w2, w5
	cmp	x3, x4
	mov	w5, w6
	b.ne	.LBB2_2
.LBB2_5:
	ret
.Lfunc_end2:
	.size	eval_At_times_u, .Lfunc_end2-eval_At_times_u
	.cfi_endproc
                                        // -- End function
	.globl	eval_AtA_times_u                // -- Begin function eval_AtA_times_u
	.p2align	2
	.type	eval_AtA_times_u,@function
eval_AtA_times_u:                       // @eval_AtA_times_u
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x3, x1
	ubfiz	x1, x2, #3, #32
	mov	x4, sp
	add	x1, x1, #15
	and	x1, x1, #0xffffffff0
	sub	x1, x4, x1
	mov	sp, x1
	cmp	w2, #1
	b.lt	.LBB3_9
// %bb.1:                               // %.preheader1
	fmov	d0, #1.00000000
	mov	x0, x2
	mov	x4, xzr
	mov	w2, w2
	mov	w5, #1                          // =0x1
.LBB3_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_3 Depth 2
	movi	d1, #0000000000000000
	add	x6, x4, #1
	mov	x7, x5
	mov	x8, x3
	mov	x9, x2
.LBB3_3:                                //   Parent Loop BB3_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	w10, w7, #1
	ldr	d3, [x8], #8
	mul	w10, w7, w10
	subs	x9, x9, #1
	add	x7, x7, #1
	add	w10, w6, w10, lsr #1
	scvtf	d2, w10
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	b.ne	.LBB3_3
// %bb.4:                               //   in Loop: Header=BB3_2 Depth=1
	cmp	x6, x2
	str	d1, [x1, x4, lsl #3]
	add	x5, x5, #1
	mov	x4, x6
	b.ne	.LBB3_2
// %bb.5:                               // %.preheader
	fmov	d0, #1.00000000
	mov	w3, wzr
	mov	x4, xzr
	mov	w5, #2                          // =0x2
.LBB3_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_7 Depth 2
	movi	d1, #0000000000000000
	add	x6, x4, #1
	mov	w7, #1                          // =0x1
	mov	w8, w3
	mov	w11, w5
	mov	x9, x2
	mov	x10, x1
.LBB3_7:                                //   Parent Loop BB3_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	w12, w7, w8, lsr #1
	ldr	d3, [x10], #8
	subs	x9, x9, #1
	add	w8, w8, w11
	add	w7, w7, #1
	scvtf	d2, w12
	add	w12, w11, #2
	mov	w11, w12
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	b.ne	.LBB3_7
// %bb.8:                               //   in Loop: Header=BB3_6 Depth=1
	add	w3, w3, w5
	add	w5, w5, #2
	cmp	x6, x2
	str	d1, [x0, x4, lsl #3]
	mov	x4, x6
	b.ne	.LBB3_6
.LBB3_9:
	mov	sp, x29
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end3:
	.size	eval_AtA_times_u, .Lfunc_end3-eval_AtA_times_u
	.cfi_endproc
                                        // -- End function
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	cmp	w0, #2
	b.ne	.LBB4_4
// %bb.1:
	ldr	x0, [x1, #8]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	strtol
	mov	x3, x0
	mov	w1, w3
	lsl	x0, x1, #3
	add	x0, x0, #15
	and	x2, x0, #0xffffffff0
	mov	x0, sp
	sub	x0, x0, x2
	mov	sp, x0
	mov	x4, sp
	sub	x2, x4, x2
	mov	sp, x2
	cmp	w3, #1
	b.lt	.LBB4_23
// %bb.2:
	cmp	x1, #3
	b.hi	.LBB4_5
.LBB4_3:
	mov	x4, xzr
	b	.LBB4_8
.LBB4_4:
	mov	x0, sp
	mov	x1, #-16000                     // =0xffffffffffffc180
	add	x0, x0, x1
	mov	sp, x0
	mov	x2, sp
	add	x2, x2, x1
	mov	sp, x2
	mov	w1, #2000                       // =0x7d0
	mov	w3, w1
	cmp	x1, #3
	b.ls	.LBB4_3
.LBB4_5:
	fmov	v0.2d, #1.00000000
	and	x4, x1, #0xfffffffc
	add	x5, x0, #16
	mov	x6, x4
.LBB4_6:                                // =>This Inner Loop Header: Depth=1
	subs	x6, x6, #4
	stp	q0, q0, [x5, #-16]
	add	x5, x5, #32
	b.ne	.LBB4_6
// %bb.7:
	cmp	x1, x4
	b.eq	.LBB4_10
.LBB4_8:                                // %.preheader3
	mov	x5, #4607182418800017408        // =0x3ff0000000000000
.LBB4_9:                                // =>This Inner Loop Header: Depth=1
	str	x5, [x0, x4, lsl #3]
	add	x4, x4, #1
	cmp	x1, x4
	b.ne	.LBB4_9
.LBB4_10:
	mov	w4, #1                          // =0x1
.LBB4_11:
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x0
	bl	eval_AtA_times_u
	mov	w0, w3
	mov	x1, x2
	mov	x2, x0
	bl	eval_AtA_times_u
	cbz	w4, .LBB4_14
// %bb.12:
	cmp	x1, #1
	b.hi	.LBB4_15
// %bb.13:
	movi	d0, #0000000000000000
	movi	d1, #0000000000000000
	mov	x3, xzr
	b	.LBB4_18
.LBB4_14:
	mov	x0, #9221120237041090560        // =0x7ff8000000000000
	fmov	d0, x0
	b	.LBB4_21
.LBB4_15:
	movi	v0.2d, #0000000000000000
	and	x3, x1, #0xfffffffe
	add	x4, x2, #8
	mov	x5, x0
	mov	x6, x3
.LBB4_16:                               // =>This Inner Loop Header: Depth=1
	ldr	q1, [x5], #16
	sub	x7, x4, #8
	subs	x6, x6, #2
	dup	v2.2d, v1.d[0]
	ld1	{ v1.d }[0], [x4]
	add	x4, x4, #16
	ld1	{ v2.d }[0], [x7]
	fmul	v1.2d, v1.2d, v1.d[0]
	fmul	v2.2d, v2.2d, v2.d[0]
	fadd	v0.2d, v0.2d, v2.2d
	fadd	v0.2d, v0.2d, v1.2d
	b.ne	.LBB4_16
// %bb.17:
	mov	d1, v0.d[1]
	cmp	x1, x3
	b.eq	.LBB4_20
.LBB4_18:                               // %.preheader
	lsl	x4, x3, #3
	sub	x1, x1, x3
	add	x2, x2, x4
	add	x0, x0, x4
.LBB4_19:                               // =>This Inner Loop Header: Depth=1
	ldr	d2, [x0], #8
	subs	x1, x1, #1
	ldr	d3, [x2], #8
	fmadd	d1, d2, d3, d1
	fmadd	d0, d3, d3, d0
	b.ne	.LBB4_19
.LBB4_20:
	fdiv	d0, d1, d0
.LBB4_21:
	fsqrt	d1, d0
	fcmp	d1, d1
	b.vs	.LBB4_24
.LBB4_22:                               // %.split
	fmov	d0, d1
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	bl	printf
	mov	w0, wzr
	mov	sp, x29
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB4_23:
	.cfi_restore_state
	mov	w4, wzr
	b	.LBB4_11
.LBB4_24:                               // %call.sqrt
	bl	sqrt
	fmov	d1, d0
	b	.LBB4_22
.Lfunc_end4:
	.size	main, .Lfunc_end4-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%0.9f\n"
	.size	.L.str, 7

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
