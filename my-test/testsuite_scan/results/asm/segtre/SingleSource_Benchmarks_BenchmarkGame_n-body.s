	.file	"n-body.c"
	.text
	.globl	advance                         // -- Begin function advance
	.p2align	2
	.type	advance,@function
advance:                                // @advance
	.cfi_startproc
// %bb.0:
	cmp	w0, #1
                                        // kill: def $d0 killed $d0 def $q0
	b.lt	.LBB0_8
// %bb.1:
	str	d10, [sp, #-32]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	d9, d8, [sp, #16]               // 16-byte Folded Spill
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -32
	mov	w1, w0
	mov	x6, xzr
	mov	w4, #56                         // =0x38
	mov	x0, x1
	mov	w1, w1
	sub	x2, x1, #1
	add	x3, x0, #104
	b	.LBB0_3
.LBB0_2:                                //   in Loop: Header=BB0_3 Depth=1
	cmp	x5, x1
	sub	x2, x2, #1
	add	x3, x3, #56
	mov	x6, x5
	b.eq	.LBB0_6
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_5 Depth 2
	add	x5, x6, #1
	cmp	x5, x1
	b.hs	.LBB0_2
// %bb.4:                               //   in Loop: Header=BB0_3 Depth=1
	madd	x6, x6, x4, x0
	mov	x7, x3
	mov	x8, x2
	ldr	q1, [x6]
	ldr	d2, [x6, #16]
	ldr	d3, [x6, #48]
.LBB0_5:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldur	q4, [x7, #-48]
	ldr	d9, [x6, #40]
	subs	x8, x8, #1
	fsub	v5.2d, v1.2d, v4.2d
	fmul	v4.2d, v5.2d, v5.2d
	mov	d6, v4.d[1]
	ldur	d4, [x7, #-32]
	fsub	d4, d2, d4
	fmadd	d6, d5, d5, d6
	fmadd	d6, d4, d4, d6
	fsqrt	d6, d6
	fmul	d7, d6, d6
	fmul	d6, d6, d7
	ldr	d7, [x7]
	fneg	d8, d7
	fnmul	d7, d7, d4
	fdiv	d6, d0, d6
	fmul	v10.2d, v5.2d, v8.d[0]
	ldur	q8, [x6, #24]
	fmul	v5.2d, v5.2d, v3.d[0]
	fmla	v8.2d, v10.2d, v6.d[0]
	fmadd	d7, d7, d6, d9
	stur	q8, [x6, #24]
	fmul	d8, d4, d3
	str	d7, [x6, #40]
	ldur	q4, [x7, #-24]
	ldur	d7, [x7, #-8]
	fmla	v4.2d, v5.2d, v6.d[0]
	fmadd	d5, d8, d6, d7
	stur	q4, [x7, #-24]
	stur	d5, [x7, #-8]
	add	x7, x7, #56
	b.ne	.LBB0_5
	b	.LBB0_2
.LBB0_6:                                // =>This Inner Loop Header: Depth=1
	ldur	q2, [x0, #24]
	ldr	q1, [x0]
	subs	x1, x1, #1
	ldr	d3, [x0, #40]
	ldr	d4, [x0, #16]
	fmla	v1.2d, v2.2d, v0.d[0]
	fmadd	d2, d0, d3, d4
	str	q1, [x0]
	str	d2, [x0, #16]
	add	x0, x0, #56
	b.ne	.LBB0_6
// %bb.7:
	ldp	d9, d8, [sp, #16]               // 16-byte Folded Reload
	ldr	d10, [sp], #32                  // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
.LBB0_8:
	ret
.Lfunc_end0:
	.size	advance, .Lfunc_end0-advance
	.cfi_endproc
                                        // -- End function
	.globl	energy                          // -- Begin function energy
	.p2align	2
	.type	energy,@function
energy:                                 // @energy
	.cfi_startproc
// %bb.0:
	cmp	w0, #1
	b.lt	.LBB1_7
// %bb.1:
	str	d8, [sp, #-16]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset b8, -16
	mov	w2, w0
	mov	x0, x1
	mov	x1, xzr
	mov	w2, w2
	movi	d0, #0000000000000000
	fmov	d1, #0.50000000
	sub	x3, x2, #1
	add	x4, x0, #56
	mov	w5, #56                         // =0x38
	b	.LBB1_3
.LBB1_2:                                //   in Loop: Header=BB1_3 Depth=1
	cmp	x1, x2
	sub	x3, x3, #1
	add	x4, x4, #56
	b.eq	.LBB1_6
.LBB1_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_5 Depth 2
	madd	x6, x1, x5, x0
	add	x1, x1, #1
	cmp	x1, x2
	ldp	d3, d2, [x6, #24]
	fmul	d2, d2, d2
	fmadd	d3, d3, d3, d2
	ldp	d5, d2, [x6, #40]
	fmul	d4, d2, d1
	fmadd	d3, d5, d5, d3
	fmadd	d0, d4, d3, d0
	b.hs	.LBB1_2
// %bb.4:                               //   in Loop: Header=BB1_3 Depth=1
	ldp	d3, d4, [x6]
	mov	x7, x3
	ldr	d5, [x6, #16]
	mov	x6, x4
.LBB1_5:                                //   Parent Loop BB1_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d6, [x6, #8]
	subs	x7, x7, #1
	fsub	d7, d4, d6
	ldr	d6, [x6]
	fsub	d6, d3, d6
	fmul	d8, d7, d7
	ldr	d7, [x6, #16]
	fsub	d7, d5, d7
	fmadd	d6, d6, d6, d8
	fmadd	d6, d7, d7, d6
	ldr	d7, [x6, #48]
	add	x6, x6, #56
	fmul	d7, d2, d7
	fsqrt	d6, d6
	fdiv	d6, d7, d6
	fsub	d0, d0, d6
	b.ne	.LBB1_5
	b	.LBB1_2
.LBB1_6:
	ldr	d8, [sp], #16                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	ret
.LBB1_7:
	movi	d0, #0000000000000000
	ret
.Lfunc_end1:
	.size	energy, .Lfunc_end1-energy
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function offset_momentum
.LCPI2_0:
	.xword	0xc043bd3cc9be45de              // double -39.478417604357432
	.text
	.globl	offset_momentum
	.p2align	2
	.type	offset_momentum,@function
offset_momentum:                        // @offset_momentum
	.cfi_startproc
// %bb.0:
	mov	w1, w0
	cmp	w0, #1
	mov	x0, x1
	b.lt	.LBB2_3
// %bb.1:
	movi	v0.2d, #0000000000000000
	movi	d1, #0000000000000000
	cmp	w1, #1
	mov	w1, w1
	b.ne	.LBB2_4
// %bb.2:
	mov	x2, xzr
	b	.LBB2_7
.LBB2_3:
	movi	v0.2d, #0000000000000000
	movi	d1, #0000000000000000
	b	.LBB2_9
.LBB2_4:
	and	x2, x1, #0x7ffffffe
	add	x3, x0, #80
	mov	x4, x2
.LBB2_5:                                // =>This Inner Loop Header: Depth=1
	ldp	d3, d2, [x3, #-40]
	subs	x4, x4, #2
	ldur	q4, [x3, #-56]
	fmul	d3, d3, d2
	fmul	v2.2d, v4.2d, v2.d[0]
	ldp	d6, d4, [x3, #16]
	ldr	q5, [x3], #112
	fadd	d1, d1, d3
	fmul	d3, d6, d4
	fmul	v4.2d, v5.2d, v4.d[0]
	fadd	v0.2d, v0.2d, v2.2d
	fadd	d1, d1, d3
	fadd	v0.2d, v0.2d, v4.2d
	b.ne	.LBB2_5
// %bb.6:
	cmp	x2, x1
	b.eq	.LBB2_9
.LBB2_7:                                // %.preheader
	mov	w3, #56                         // =0x38
	sub	x1, x1, x2
	umaddl	x3, w2, w3, x0
	add	x2, x3, #48
.LBB2_8:                                // =>This Inner Loop Header: Depth=1
	ldp	d4, d2, [x2, #-8]
	subs	x1, x1, #1
	ldur	q3, [x2, #-24]
	add	x2, x2, #56
	fmla	v0.2d, v3.2d, v2.d[0]
	fmadd	d1, d4, d2, d1
	b.ne	.LBB2_8
.LBB2_9:
	mov	x1, #17886                      // =0x45de
	movk	x1, #51646, lsl #16
	movk	x1, #48444, lsl #32
	movk	x1, #49219, lsl #48
	dup	v2.2d, x1
	adrp	x1, .LCPI2_0
	fdiv	v0.2d, v0.2d, v2.2d
	ldr	d2, [x1, :lo12:.LCPI2_0]
	fdiv	d1, d1, d2
	stur	q0, [x0, #24]
	str	d1, [x0, #40]
	ret
.Lfunc_end2:
	.size	offset_momentum, .Lfunc_end2-offset_momentum
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function main
.LCPI3_0:
	.xword	0xc043bd3cc9be45de              // double -39.478417604357432
.LCPI3_1:
	.xword	0x3f847ae147ae147b              // double 0.01
	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	d15, d14, [sp, #16]             // 16-byte Folded Spill
	stp	d13, d12, [sp, #32]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #48]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #64]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #80]             // 16-byte Folded Spill
	add	x29, sp, #80
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -24
	.cfi_offset b9, -32
	.cfi_offset b10, -40
	.cfi_offset b11, -48
	.cfi_offset b12, -56
	.cfi_offset b13, -64
	.cfi_offset b14, -72
	.cfi_offset b15, -80
	adrp	x0, bodies
	add	x0, x0, :lo12:bodies
	movi	v17.2d, #0000000000000000
	ldp	d5, d3, [x0]
	ldr	d4, [x0, #16]
	ldp	d1, d0, [x0, #56]
	ldr	d11, [x0, #112]
	ldp	d21, d26, [x0, #40]
	movi	d27, #0000000000000000
	fsub	d10, d5, d11
	ldp	d24, d23, [x0, #96]
	fsub	d2, d3, d0
	fsub	d6, d5, d1
	ldr	q25, [x0, #80]
	ldr	d28, [x0, #240]
	ldur	q22, [x0, #136]
	ldr	d20, [x0, #152]
	fmadd	d21, d21, d26, d27
	ldr	q18, [x0, #192]
	mov	x1, #17886                      // =0x45de
	ldur	q15, [x0, #248]
	movk	x1, #51646, lsl #16
	ldur	q30, [x0, #168]
	fmul	d7, d2, d2
	ldr	d2, [x0, #72]
	movk	x1, #48444, lsl #32
	movk	x1, #49219, lsl #48
	fmul	d31, d26, d23
	dup	v5.2d, v5.d[0]
	fsub	d8, d4, d2
	fmadd	d21, d24, d23, d21
	dup	v29.2d, x1
	adrp	x1, .LCPI3_0
	fmadd	d6, d6, d6, d7
	fmadd	d6, d8, d8, d6
	ldp	d8, d7, [x0, #112]
	fsub	d9, d3, d7
	dup	v3.2d, v3.d[0]
	fsqrt	d6, d6
	fmul	d13, d9, d9
	ldp	d9, d12, [x0, #120]
	fsub	d14, d4, d12
	dup	v4.2d, v4.d[0]
	fmadd	d10, d10, d10, d13
	fsub	d13, d0, d7
	fsub	d7, d1, d11
	fsub	d11, d2, d12
	dup	v1.2d, v1.d[0]
	dup	v2.2d, v2.d[0]
	fmul	d13, d13, d13
	fmadd	d10, d14, d14, d10
	fmadd	d7, d7, d7, d13
	ldp	d13, d12, [x0, #224]
	fmadd	d7, d11, d11, d7
	ldr	d11, [x0, #176]
	fsub	d12, d11, d12
	ldp	d19, d11, [x0, #160]
	fsub	d11, d11, d13
	ldr	d13, [x0, #184]
	fmadd	d21, d20, d19, d21
	fmul	d12, d12, d12
	fsub	d13, d13, d28
	fmadd	d11, d11, d11, d12
	ldur	q12, [x0, #24]
	fdiv	d6, d31, d6
	ldur	q31, [x0, #184]
	fmla	v17.2d, v12.2d, v26.d[0]
	ldp	d14, d12, [x0, #264]
	mov	v31.d[1], v28.d[0]
	fmadd	d11, d13, d13, d11
	ldp	d13, d16, [x0, #208]
	fmla	v17.2d, v25.2d, v23.d[0]
	fmadd	d21, d13, d16, d21
	fsub	v4.2d, v4.2d, v31.2d
	fmla	v17.2d, v22.2d, v19.d[0]
	fmadd	d21, d14, d12, d21
	fmla	v17.2d, v18.2d, v16.d[0]
	fmla	v17.2d, v15.2d, v12.d[0]
	fsqrt	d10, d10
	fdiv	v17.2d, v17.2d, v29.2d
	ldr	d29, [x1, :lo12:.LCPI3_0]
	fdiv	d21, d21, d29
	ldr	q29, [x0, #224]
	str	q29, [sp]                       // 16-byte Folded Spill
	ldr	q0, [sp]                        // 16-byte Folded Reload
	stur	q17, [x0, #24]
	zip2	v29.2d, v30.2d, v0.2d
	ldr	q0, [sp]                        // 16-byte Folded Reload
	zip1	v30.2d, v30.2d, v0.2d
	dup	v0.2d, v0.d[0]
	fsub	v3.2d, v3.2d, v29.2d
	fsub	v0.2d, v0.2d, v29.2d
	fsub	v5.2d, v5.2d, v30.2d
	fsub	v1.2d, v1.2d, v30.2d
	fmul	v3.2d, v3.2d, v3.2d
	fmul	v0.2d, v0.2d, v0.2d
	fmla	v3.2d, v5.2d, v5.2d
	fmla	v0.2d, v1.2d, v1.2d
	fsub	v1.2d, v2.2d, v31.2d
	dup	v2.2d, v8.d[0]
	fsqrt	d7, d7
	fmla	v3.2d, v4.2d, v4.2d
	fmul	d4, d26, d19
	fsub	v2.2d, v2.2d, v30.2d
	fmla	v0.2d, v1.2d, v1.2d
	dup	v1.2d, v9.d[0]
	str	d21, [x0, #40]
	fsub	v1.2d, v1.2d, v29.2d
	fmul	v1.2d, v1.2d, v1.2d
	fmla	v1.2d, v2.2d, v2.2d
	fmul	v2.2d, v17.2d, v17.2d
	fsqrt	v3.2d, v3.2d
	fdiv	d4, d4, d10
	mov	v10.16b, v16.16b
	mov	v10.d[1], v12.d[0]
	fmul	v5.2d, v10.2d, v26.d[0]
	fmul	v9.2d, v10.2d, v19.d[0]
	fsqrt	v0.2d, v0.2d
	fdiv	v5.2d, v5.2d, v3.2d
	fmul	d3, d23, d19
	fdiv	d3, d3, d7
	ldr	d7, [x0, #128]
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	dup	v7.2d, v7.d[0]
	fsub	v7.2d, v7.2d, v31.2d
	fmla	v1.2d, v7.2d, v7.2d
	mov	d7, v2.d[1]
	fmul	v2.2d, v10.2d, v23.d[0]
	fmadd	d7, d17, d17, d7
	fmadd	d7, d21, d21, d7
	fdiv	v2.2d, v2.2d, v0.2d
	fmov	d0, #0.50000000
	fmul	d8, d26, d0
	fmadd	d8, d8, d7, d27
	fmul	v7.2d, v25.2d, v25.2d
	fsub	d8, d8, d6
	mov	d6, v7.d[1]
	fsub	d4, d8, d4
	fmadd	d6, d25, d25, d6
	mov	d8, v5.d[1]
	fsqrt	v1.2d, v1.2d
	fsub	d7, d4, d5
	fmul	d4, d23, d0
	fmadd	d5, d24, d24, d6
	fmul	v6.2d, v22.2d, v22.2d
	fsub	d7, d7, d8
	fmadd	d5, d4, d5, d7
	mov	d4, v6.d[1]
	fmul	d6, d16, d12
	fmul	v7.2d, v18.2d, v18.2d
	fsub	d3, d5, d3
	fmadd	d4, d22, d22, d4
	mov	d5, v2.d[1]
	fsub	d3, d3, d2
	fmadd	d8, d20, d20, d4
	mov	d4, v7.d[1]
	fsqrt	d11, d11
	fsub	d3, d3, d5
	fmadd	d5, d18, d18, d4
	fdiv	v1.2d, v9.2d, v1.2d
	fdiv	d2, d6, d11
	fmul	d6, d19, d0
	fmadd	d6, d6, d8, d3
	fmul	v3.2d, v15.2d, v15.2d
	fsub	d4, d6, d1
	mov	d6, v1.d[1]
	fmul	d1, d16, d0
	mov	d7, v3.d[1]
	fmadd	d3, d13, d13, d5
	fmul	d0, d12, d0
	fsub	d5, d4, d6
	fmadd	d4, d15, d15, d7
	fmadd	d1, d1, d3, d5
	fmadd	d3, d14, d14, d4
	fsub	d1, d1, d2
	fmadd	d0, d0, d3, d1
	bl	printf
	adrp	x1, .LCPI3_1
	ldr	d0, [x1, :lo12:.LCPI3_1]
	mov	w1, #19264                      // =0x4b40
	movk	w1, #76, lsl #16
.LBB3_1:                                // =>This Inner Loop Header: Depth=1
	mov	w0, #5                          // =0x5
	mov	x1, x0
	bl	advance
	subs	w1, w1, #1
	b.ne	.LBB3_1
// %bb.2:
	ldp	d4, d5, [x0]
	ldr	d3, [x0, #16]
	ldp	d1, d2, [x0, #56]
	ldr	d10, [x0, #112]
	ldr	d15, [x0, #240]
	ldur	q16, [x0, #168]
	ldr	q19, [x0, #224]
	fsub	d11, d4, d10
	fsub	d0, d5, d2
	fsub	d6, d4, d1
	zip2	v17.2d, v16.2d, v19.2d
	dup	v4.2d, v4.d[0]
	fmul	d7, d0, d0
	ldr	d0, [x0, #72]
	fsub	d8, d3, d0
	fmadd	d6, d6, d6, d7
	fmadd	d6, d8, d8, d6
	ldp	d8, d7, [x0, #112]
	fsub	d9, d5, d7
	dup	v5.2d, v5.d[0]
	fsqrt	d6, d6
	fsub	v5.2d, v5.2d, v17.2d
	fmul	d13, d9, d9
	ldp	d9, d12, [x0, #120]
	fsub	d14, d3, d12
	dup	v3.2d, v3.d[0]
	fmadd	d11, d11, d11, d13
	fsub	d13, d2, d7
	fsub	d7, d1, d10
	fsub	d10, d0, d12
	dup	v2.2d, v2.d[0]
	dup	v1.2d, v1.d[0]
	dup	v0.2d, v0.d[0]
	fmul	d13, d13, d13
	fmadd	d11, d14, d14, d11
	ldr	d14, [x0, #48]
	fmadd	d7, d7, d7, d13
	ldp	d13, d12, [x0, #224]
	fmadd	d7, d10, d10, d7
	ldr	d10, [x0, #176]
	fsub	d12, d10, d12
	ldr	d10, [x0, #168]
	fsub	d10, d10, d13
	ldr	d13, [x0, #184]
	fmul	d12, d12, d12
	fsub	d13, d13, d15
	fmadd	d10, d10, d10, d12
	fsqrt	d11, d11
	fmadd	d10, d13, d13, d10
	ldp	d12, d13, [x0, #96]
	fmul	d18, d14, d13
	fdiv	d6, d18, d6
	zip1	v18.2d, v16.2d, v19.2d
	ldur	q16, [x0, #184]
	ldur	q19, [x0, #216]
	mov	v16.d[1], v15.d[0]
	mov	v21.16b, v19.16b
	mov	v19.d[1], v13.d[0]
	fsub	v15.2d, v4.2d, v18.2d
	fmul	v4.2d, v5.2d, v5.2d
	ldr	d5, [x0, #160]
	fsub	v3.2d, v3.2d, v16.2d
	mov	v21.d[1], v14.d[0]
	fsub	v0.2d, v0.2d, v16.2d
	fmla	v4.2d, v15.2d, v15.2d
	fmla	v4.2d, v3.2d, v3.2d
	fmul	d3, d14, d5
	fsqrt	v20.2d, v4.2d
	ldr	d4, [x0, #272]
	fsqrt	d7, d7
	fdiv	d11, d3, d11
	fmov	d3, #0.50000000
	fmul	d15, d14, d3
	mov	v14.d[1], v4.d[0]
	fmul	v14.2d, v21.2d, v14.2d
	fdiv	v14.2d, v14.2d, v20.2d
	fmul	d20, d13, d5
	fdiv	d7, d20, d7
	fsub	v20.2d, v2.2d, v17.2d
	fsub	v2.2d, v1.2d, v18.2d
	fmul	v1.2d, v20.2d, v20.2d
	fmla	v1.2d, v2.2d, v2.2d
	dup	v2.2d, v8.d[0]
	ldr	d8, [x0, #128]
	dup	v8.2d, v8.d[0]
	fsub	v2.2d, v2.2d, v18.2d
	fmla	v1.2d, v0.2d, v0.2d
	fsub	v8.2d, v8.2d, v16.2d
	fsqrt	v0.2d, v1.2d
	dup	v1.2d, v9.d[0]
	fsub	v1.2d, v1.2d, v17.2d
	fmul	v1.2d, v1.2d, v1.2d
	fmla	v1.2d, v2.2d, v2.2d
	fmul	d2, d13, d3
	mov	v13.d[1], v4.d[0]
	fmla	v1.2d, v8.2d, v8.2d
	fmul	v13.2d, v19.2d, v13.2d
	fsqrt	v8.2d, v1.2d
	ldp	d9, d1, [x0, #24]
	fmul	d1, d1, d1
	fmadd	d9, d9, d9, d1
	fdiv	v1.2d, v13.2d, v0.2d
	ldr	d0, [x0, #40]
	movi	d13, #0000000000000000
	fmadd	d9, d0, d0, d9
	ldr	d0, [x0, #216]
	fmadd	d13, d15, d9, d13
	mov	v9.16b, v0.16b
	ldr	d15, [x0, #88]
	mov	v9.d[1], v4.d[0]
	fsub	d6, d13, d6
	fmul	v13.2d, v9.2d, v5.d[0]
	fmul	d9, d15, d15
	ldr	d15, [x0, #80]
	fsub	d11, d6, d11
	fmul	d5, d5, d3
	fmadd	d9, d15, d15, d9
	fsqrt	d10, d10
	fmadd	d9, d12, d12, d9
	fdiv	v6.2d, v13.2d, v8.2d
	fsub	d8, d11, d14
	mov	d11, v14.d[1]
	fsub	d8, d8, d11
	fmadd	d8, d2, d9, d8
	ldp	d2, d11, [x0, #144]
	fmul	d9, d0, d4
	fmul	d0, d0, d3
	fmul	d2, d2, d2
	fsub	d8, d8, d7
	ldr	d7, [x0, #136]
	fmadd	d7, d7, d7, d2
	fsub	d2, d8, d1
	mov	d8, v1.d[1]
	fdiv	d1, d9, d10
	fmadd	d7, d11, d11, d7
	fsub	d2, d2, d8
	ldr	d8, [x0, #200]
	fmul	d8, d8, d8
	fmadd	d5, d5, d7, d2
	ldr	d2, [x0, #192]
	ldr	d7, [x0, #208]
	fmadd	d2, d2, d2, d8
	ldr	d8, [x0, #256]
	fsub	d5, d5, d6
	mov	d6, v6.d[1]
	fmul	d8, d8, d8
	fmadd	d2, d7, d7, d2
	fsub	d6, d5, d6
	ldr	d5, [x0, #248]
	fmadd	d5, d5, d5, d8
	fmadd	d0, d0, d2, d6
	ldr	d6, [x0, #264]
	fmul	d2, d4, d3
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	fmadd	d3, d6, d6, d5
	fsub	d0, d0, d1
	fmadd	d0, d2, d3, d0
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 96
	ldp	x29, x30, [sp, #80]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #64]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #48]             // 16-byte Folded Reload
	ldp	d13, d12, [sp, #32]             // 16-byte Folded Reload
	ldp	d15, d14, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #96
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
	.cfi_restore b14
	.cfi_restore b15
	ret
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
	.cfi_endproc
                                        // -- End function
	.type	bodies,@object                  // @bodies
	.data
	.globl	bodies
	.p2align	3, 0x0
bodies:
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x4043bd3cc9be45de              // double 39.478417604357432
	.xword	0x40135da0343cd92c              // double 4.8414314424647209
	.xword	0xbff290abc01fdb7c              // double -1.1603200440274284
	.xword	0xbfba86f96c25ebf0              // double -0.10362204447112311
	.xword	0x3fe367069b93ccbc              // double 0.60632639299583202
	.xword	0x40067ef2f57d949b              // double 2.8119868449162602
	.xword	0xbf99d2d79a5a0715              // double -0.025218361659887629
	.xword	0x3fa34c95d9ab33d8              // double 0.037693674870389493
	.xword	0x4020afcdc332ca67              // double 8.3433667182445799
	.xword	0x40107fcb31de01b0              // double 4.1247985641243048
	.xword	0xbfd9d353e1eb467c              // double -0.40352341711432138
	.xword	0xbff02c21b8879442              // double -1.0107743461787924
	.xword	0x3ffd35e9bf1f8f13              // double 1.8256623712304119
	.xword	0x3f813c485f1123b4              // double 0.0084157613765841535
	.xword	0x3f871d490d07c637              // double 0.011286326131968767
	.xword	0x4029c9eacea7d9cf              // double 12.894369562139131
	.xword	0xc02e38e8d626667e              // double -15.111151401698631
	.xword	0xbfcc9557be257da0              // double -0.22330757889265573
	.xword	0x3ff1531ca9911bef              // double 1.0827910064415354
	.xword	0x3febcc7f3e54bbc5              // double 0.86871301816960822
	.xword	0xbf862f6bfaf23e7c              // double -0.010832637401363636
	.xword	0x3f5c3dd29cf41eb3              // double 0.0017237240570597112
	.xword	0x402ec267a905572a              // double 15.379697114850917
	.xword	0xc039eb5833c8a220              // double -25.919314609987964
	.xword	0x3fc6f1f393abe540              // double 0.17925877295037118
	.xword	0x3fef54b61659bc4a              // double 0.97909073224389798
	.xword	0x3fe307c631c4fba3              // double 0.59469899864767617
	.xword	0xbfa1cb88587665f6              // double -0.034755955504078104
	.xword	0x3f60a8f3531799ac              // double 0.0020336868699246304
	.size	bodies, 280

	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%.9f\n"
	.size	.L.str, 6

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
