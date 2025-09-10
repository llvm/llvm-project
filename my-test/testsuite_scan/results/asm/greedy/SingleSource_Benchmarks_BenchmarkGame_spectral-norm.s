	.file	"spectral-norm.c"
	.text
	.globl	eval_A                          // -- Begin function eval_A
	.p2align	2
	.type	eval_A,@function
eval_A:                                 // @eval_A
	.cfi_startproc
// %bb.0:
	add	w8, w1, w0
	fmov	d1, #1.00000000
	madd	w8, w8, w8, w8
	add	w8, w8, w8, lsr #31
	add	w8, w0, w8, asr #1
	add	w8, w8, #1
	scvtf	d0, w8
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
	cmp	w0, #1
	b.lt	.LBB1_5
// %bb.1:
	fmov	d0, #1.00000000
	mov	x8, xzr
	mov	w9, w0
	mov	w10, #1                         // =0x1
.LBB1_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_3 Depth 2
	movi	d1, #0000000000000000
	add	x11, x8, #1
	mov	x12, x10
	mov	x13, x1
	mov	x14, x9
	str	xzr, [x2, x8, lsl #3]
.LBB1_3:                                //   Parent Loop BB1_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	w15, w12, #1
	ldr	d3, [x13], #8
	mul	w15, w12, w15
	subs	x14, x14, #1
	add	x12, x12, #1
	add	w15, w11, w15, lsr #1
	scvtf	d2, w15
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	str	d1, [x2, x8, lsl #3]
	b.ne	.LBB1_3
// %bb.4:                               //   in Loop: Header=BB1_2 Depth=1
	cmp	x11, x9
	add	x10, x10, #1
	mov	x8, x11
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
	cmp	w0, #1
	b.lt	.LBB2_5
// %bb.1:
	fmov	d0, #1.00000000
	mov	w8, wzr
	mov	x9, xzr
	mov	w10, w0
	mov	w11, #2                         // =0x2
.LBB2_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB2_3 Depth 2
	movi	d1, #0000000000000000
	mov	w12, #1                         // =0x1
	mov	w13, w8
	mov	w16, w11
	mov	x14, x10
	mov	x15, x1
	str	xzr, [x2, x9, lsl #3]
.LBB2_3:                                //   Parent Loop BB2_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	w17, w12, w13, lsr #1
	ldr	d3, [x15], #8
	subs	x14, x14, #1
	add	w13, w13, w16
	add	w16, w16, #2
	scvtf	d2, w17
	add	w12, w12, #1
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	str	d1, [x2, x9, lsl #3]
	b.ne	.LBB2_3
// %bb.4:                               //   in Loop: Header=BB2_2 Depth=1
	add	x9, x9, #1
	add	w12, w11, #2
	add	w8, w8, w11
	cmp	x9, x10
	mov	w11, w12
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
                                        // kill: def $w0 killed $w0 def $x0
	ubfiz	x8, x0, #3, #32
	mov	x9, sp
	add	x8, x8, #15
	and	x8, x8, #0xffffffff0
	sub	x8, x9, x8
	mov	sp, x8
	cmp	w0, #1
	b.lt	.LBB3_9
// %bb.1:                               // %.preheader1
	fmov	d0, #1.00000000
	mov	x10, xzr
	mov	w9, w0
	mov	w11, #1                         // =0x1
.LBB3_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_3 Depth 2
	movi	d1, #0000000000000000
	add	x12, x10, #1
	mov	x13, x11
	mov	x14, x1
	mov	x15, x9
.LBB3_3:                                //   Parent Loop BB3_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	w16, w13, #1
	ldr	d3, [x14], #8
	mul	w16, w13, w16
	subs	x15, x15, #1
	add	x13, x13, #1
	add	w16, w12, w16, lsr #1
	scvtf	d2, w16
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	b.ne	.LBB3_3
// %bb.4:                               //   in Loop: Header=BB3_2 Depth=1
	cmp	x12, x9
	str	d1, [x8, x10, lsl #3]
	add	x11, x11, #1
	mov	x10, x12
	b.ne	.LBB3_2
// %bb.5:                               // %.preheader
	fmov	d0, #1.00000000
	mov	w10, wzr
	mov	x11, xzr
	mov	w12, #2                         // =0x2
.LBB3_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_7 Depth 2
	movi	d1, #0000000000000000
	add	x13, x11, #1
	mov	w14, #1                         // =0x1
	mov	w15, w10
	mov	w18, w12
	mov	x16, x9
	mov	x17, x8
.LBB3_7:                                //   Parent Loop BB3_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	w0, w14, w15, lsr #1
	ldr	d3, [x17], #8
	subs	x16, x16, #1
	add	w15, w15, w18
	add	w14, w14, #1
	scvtf	d2, w0
	add	w0, w18, #2
	mov	w18, w0
	fdiv	d2, d0, d2
	fmadd	d1, d2, d3, d1
	b.ne	.LBB3_7
// %bb.8:                               //   in Loop: Header=BB3_6 Depth=1
	add	w10, w10, w12
	add	w12, w12, #2
	cmp	x13, x9
	str	d1, [x2, x11, lsl #3]
	mov	x11, x13
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
	stp	x29, x30, [sp, #-64]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	str	x23, [sp, #16]                  // 8-byte Folded Spill
	stp	x22, x21, [sp, #32]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_remember_state
	cmp	w0, #2
	b.ne	.LBB4_4
// %bb.1:
	ldr	x0, [x1, #8]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	strtol
	mov	x21, x0
	mov	x9, sp
	mov	w22, w21
	lsl	x8, x22, #3
	add	x8, x8, #15
	and	x8, x8, #0xffffffff0
	sub	x19, x9, x8
	mov	sp, x19
	mov	x9, sp
	sub	x20, x9, x8
	mov	sp, x20
	cmp	w21, #1
	b.lt	.LBB4_23
// %bb.2:
	cmp	x22, #3
	b.hi	.LBB4_5
.LBB4_3:
	mov	x8, xzr
	b	.LBB4_8
.LBB4_4:
	mov	x8, sp
	mov	x9, #-16000                     // =0xffffffffffffc180
	add	x19, x8, x9
	mov	sp, x19
	mov	x8, sp
	add	x20, x8, x9
	mov	sp, x20
	mov	w22, #2000                      // =0x7d0
	mov	w21, w22
	cmp	x22, #3
	b.ls	.LBB4_3
.LBB4_5:
	fmov	v0.2d, #1.00000000
	and	x8, x22, #0xfffffffc
	add	x9, x19, #16
	mov	x10, x8
.LBB4_6:                                // =>This Inner Loop Header: Depth=1
	subs	x10, x10, #4
	stp	q0, q0, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB4_6
// %bb.7:
	cmp	x22, x8
	b.eq	.LBB4_10
.LBB4_8:                                // %.preheader3
	mov	x9, #4607182418800017408        // =0x3ff0000000000000
.LBB4_9:                                // =>This Inner Loop Header: Depth=1
	str	x9, [x19, x8, lsl #3]
	add	x8, x8, #1
	cmp	x22, x8
	b.ne	.LBB4_9
.LBB4_10:
	mov	w23, #1                         // =0x1
.LBB4_11:
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x19
	mov	x2, x20
	bl	eval_AtA_times_u
	mov	w0, w21
	mov	x1, x20
	mov	x2, x19
	bl	eval_AtA_times_u
	cbz	w23, .LBB4_14
// %bb.12:
	cmp	x22, #1
	b.hi	.LBB4_15
// %bb.13:
	movi	d0, #0000000000000000
	movi	d1, #0000000000000000
	mov	x8, xzr
	b	.LBB4_18
.LBB4_14:
	mov	x8, #9221120237041090560        // =0x7ff8000000000000
	fmov	d0, x8
	b	.LBB4_21
.LBB4_15:
	movi	v0.2d, #0000000000000000
	and	x8, x22, #0xfffffffe
	add	x9, x20, #8
	mov	x10, x19
	mov	x11, x8
.LBB4_16:                               // =>This Inner Loop Header: Depth=1
	ldr	q1, [x10], #16
	sub	x12, x9, #8
	subs	x11, x11, #2
	dup	v2.2d, v1.d[0]
	ld1	{ v1.d }[0], [x9]
	add	x9, x9, #16
	ld1	{ v2.d }[0], [x12]
	fmul	v1.2d, v1.2d, v1.d[0]
	fmul	v2.2d, v2.2d, v2.d[0]
	fadd	v0.2d, v0.2d, v2.2d
	fadd	v0.2d, v0.2d, v1.2d
	b.ne	.LBB4_16
// %bb.17:
	mov	d1, v0.d[1]
	cmp	x22, x8
	b.eq	.LBB4_20
.LBB4_18:                               // %.preheader
	lsl	x10, x8, #3
	sub	x8, x22, x8
	add	x9, x20, x10
	add	x10, x19, x10
.LBB4_19:                               // =>This Inner Loop Header: Depth=1
	ldr	d2, [x10], #8
	subs	x8, x8, #1
	ldr	d3, [x9], #8
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
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldr	x23, [sp, #16]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #64             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB4_23:
	.cfi_restore_state
	mov	w23, wzr
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
