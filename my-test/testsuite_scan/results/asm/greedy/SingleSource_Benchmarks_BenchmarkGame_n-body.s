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
	b.lt	.LBB0_7
// %bb.1:
	mov	w8, w0
	mov	x13, xzr
	add	x10, x1, #104
	sub	x9, x8, #1
	mov	w11, #56                        // =0x38
	b	.LBB0_3
.LBB0_2:                                //   in Loop: Header=BB0_3 Depth=1
	cmp	x12, x8
	sub	x9, x9, #1
	add	x10, x10, #56
	mov	x13, x12
	b.eq	.LBB0_6
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_5 Depth 2
	add	x12, x13, #1
	cmp	x12, x8
	b.hs	.LBB0_2
// %bb.4:                               //   in Loop: Header=BB0_3 Depth=1
	madd	x13, x13, x11, x1
	mov	x14, x10
	mov	x15, x9
	ldr	q1, [x13]
	ldr	d2, [x13, #16]
	ldr	d3, [x13, #48]
.LBB0_5:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldur	q4, [x14, #-48]
	ldur	d6, [x14, #-32]
	subs	x15, x15, #1
	ldur	q17, [x13, #24]
	ldr	d18, [x13, #40]
	fsub	v4.2d, v1.2d, v4.2d
	fsub	d6, d2, d6
	fmul	v5.2d, v4.2d, v4.2d
	mov	d5, v5.d[1]
	fmadd	d5, d4, d4, d5
	fmadd	d5, d6, d6, d5
	fsqrt	d5, d5
	fmul	d7, d5, d5
	fmul	d5, d5, d7
	ldr	d7, [x14]
	fneg	d16, d7
	fnmul	d7, d7, d6
	fmul	d6, d6, d3
	fdiv	d5, d0, d5
	fmul	v16.2d, v4.2d, v16.d[0]
	fmul	v4.2d, v4.2d, v3.d[0]
	fmla	v17.2d, v16.2d, v5.d[0]
	fmadd	d7, d7, d5, d18
	stur	q17, [x13, #24]
	str	d7, [x13, #40]
	ldur	q7, [x14, #-24]
	ldur	d16, [x14, #-8]
	fmla	v7.2d, v4.2d, v5.d[0]
	fmadd	d4, d6, d5, d16
	stur	q7, [x14, #-24]
	stur	d4, [x14, #-8]
	add	x14, x14, #56
	b.ne	.LBB0_5
	b	.LBB0_2
.LBB0_6:                                // =>This Inner Loop Header: Depth=1
	ldur	q1, [x1, #24]
	ldr	q2, [x1]
	subs	x8, x8, #1
	ldr	d3, [x1, #40]
	ldr	d4, [x1, #16]
	fmla	v2.2d, v1.2d, v0.d[0]
	fmadd	d1, d0, d3, d4
	str	q2, [x1]
	str	d1, [x1, #16]
	add	x1, x1, #56
	b.ne	.LBB0_6
.LBB0_7:
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
	b.lt	.LBB1_6
// %bb.1:
	mov	w9, w0
	mov	x8, xzr
	movi	d0, #0000000000000000
	fmov	d1, #0.50000000
	sub	x10, x9, #1
	add	x11, x1, #56
	mov	w12, #56                        // =0x38
	b	.LBB1_3
.LBB1_2:                                //   in Loop: Header=BB1_3 Depth=1
	cmp	x8, x9
	sub	x10, x10, #1
	add	x11, x11, #56
	b.eq	.LBB1_7
.LBB1_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB1_5 Depth 2
	madd	x13, x8, x12, x1
	add	x8, x8, #1
	cmp	x8, x9
	ldp	d3, d2, [x13, #24]
	fmul	d2, d2, d2
	fmadd	d3, d3, d3, d2
	ldp	d4, d2, [x13, #40]
	fmul	d5, d2, d1
	fmadd	d3, d4, d4, d3
	fmadd	d0, d5, d3, d0
	b.hs	.LBB1_2
// %bb.4:                               //   in Loop: Header=BB1_3 Depth=1
	ldp	d3, d4, [x13]
	mov	x14, x10
	ldr	d5, [x13, #16]
	mov	x13, x11
.LBB1_5:                                //   Parent Loop BB1_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	d7, d6, [x13]
	subs	x14, x14, #1
	ldr	d16, [x13, #16]
	fsub	d6, d4, d6
	fsub	d7, d3, d7
	fsub	d16, d5, d16
	fmul	d6, d6, d6
	fmadd	d6, d7, d7, d6
	ldr	d7, [x13, #48]
	add	x13, x13, #56
	fmul	d7, d2, d7
	fmadd	d6, d16, d16, d6
	fsqrt	d6, d6
	fdiv	d6, d7, d6
	fsub	d0, d0, d6
	b.ne	.LBB1_5
	b	.LBB1_2
.LBB1_6:
	movi	d0, #0000000000000000
.LBB1_7:
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
	cmp	w0, #1
	b.lt	.LBB2_3
// %bb.1:
	movi	v0.2d, #0000000000000000
	movi	d1, #0000000000000000
	cmp	w0, #1
	mov	w8, w0
	b.ne	.LBB2_4
// %bb.2:
	mov	x9, xzr
	b	.LBB2_7
.LBB2_3:
	movi	v0.2d, #0000000000000000
	movi	d1, #0000000000000000
	b	.LBB2_9
.LBB2_4:
	and	x9, x8, #0x7ffffffe
	add	x10, x1, #80
	mov	x11, x9
.LBB2_5:                                // =>This Inner Loop Header: Depth=1
	ldp	d3, d2, [x10, #-40]
	subs	x11, x11, #2
	ldur	q4, [x10, #-56]
	fmul	d3, d3, d2
	fmul	v2.2d, v4.2d, v2.d[0]
	ldp	d5, d4, [x10, #16]
	ldr	q6, [x10], #112
	fadd	d1, d1, d3
	fmul	d3, d5, d4
	fmul	v4.2d, v6.2d, v4.d[0]
	fadd	v0.2d, v0.2d, v2.2d
	fadd	d1, d1, d3
	fadd	v0.2d, v0.2d, v4.2d
	b.ne	.LBB2_5
// %bb.6:
	cmp	x9, x8
	b.eq	.LBB2_9
.LBB2_7:                                // %.preheader
	mov	w10, #56                        // =0x38
	sub	x8, x8, x9
	umaddl	x10, w9, w10, x1
	add	x9, x10, #48
.LBB2_8:                                // =>This Inner Loop Header: Depth=1
	ldp	d4, d3, [x9, #-8]
	subs	x8, x8, #1
	ldur	q2, [x9, #-24]
	add	x9, x9, #56
	fmla	v0.2d, v2.2d, v3.d[0]
	fmadd	d1, d4, d3, d1
	b.ne	.LBB2_8
.LBB2_9:
	mov	x8, #17886                      // =0x45de
	movk	x8, #51646, lsl #16
	movk	x8, #48444, lsl #32
	movk	x8, #49219, lsl #48
	dup	v2.2d, x8
	adrp	x8, .LCPI2_0
	fdiv	v0.2d, v0.2d, v2.2d
	ldr	d2, [x8, :lo12:.LCPI2_0]
	fdiv	d1, d1, d2
	stur	q0, [x1, #24]
	str	d1, [x1, #40]
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
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	d15, d14, [sp, #32]             // 16-byte Folded Spill
	stp	d13, d12, [sp, #48]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #64]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #80]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #96]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            // 16-byte Folded Spill
	add	x29, sp, #96
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	.cfi_offset b10, -56
	.cfi_offset b11, -64
	.cfi_offset b12, -72
	.cfi_offset b13, -80
	.cfi_offset b14, -88
	.cfi_offset b15, -96
	adrp	x19, bodies
	add	x19, x19, :lo12:bodies
	movi	v23.2d, #0000000000000000
	ldp	d4, d5, [x19]
	ldr	d3, [x19, #16]
	ldp	d1, d27, [x19, #56]
	ldr	d18, [x19, #112]
	ldp	d28, d10, [x19, #40]
	movi	d11, #0000000000000000
	fsub	d19, d4, d18
	ldp	d31, d8, [x19, #96]
	fsub	d2, d5, d27
	fsub	d6, d4, d1
	fsub	d18, d1, d18
	ldr	q9, [x19, #80]
	ldr	d12, [x19, #240]
	ldur	q14, [x19, #168]
	ldr	q15, [x19, #224]
	ldur	q30, [x19, #136]
	ldr	q24, [x19, #192]
	dup	v4.2d, v4.d[0]
	fmadd	d28, d28, d10, d11
	mov	x8, #17886                      // =0x45de
	fmul	d7, d2, d2
	ldr	d2, [x19, #72]
	movk	x8, #51646, lsl #16
	dup	v1.2d, v1.d[0]
	movk	x8, #48444, lsl #32
	ldr	d25, [x19, #152]
	fsub	d16, d3, d2
	movk	x8, #49219, lsl #48
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	fmadd	d28, d31, d8, d28
	dup	v13.2d, x8
	fmadd	d6, d6, d6, d7
	adrp	x8, .LCPI3_0
	fmadd	d6, d16, d16, d6
	fsqrt	d0, d6
	str	d0, [sp, #24]                   // 8-byte Folded Spill
	ldp	d0, d17, [x19, #112]
	ldr	d7, [sp, #24]                   // 8-byte Folded Reload
	fsub	d16, d5, d17
	fsub	d17, d27, d17
	str	q0, [sp]                        // 16-byte Folded Spill
	zip2	v0.2d, v14.2d, v15.2d
	dup	v5.2d, v5.d[0]
	zip1	v14.2d, v14.2d, v15.2d
	dup	v27.2d, v27.d[0]
	ldur	q15, [x19, #184]
	fmul	d20, d16, d16
	ldp	d16, d21, [x19, #120]
	fmul	d17, d17, d17
	fsub	v5.2d, v5.2d, v0.2d
	fsub	v27.2d, v27.2d, v0.2d
	mov	v15.d[1], v12.d[0]
	fsub	v4.2d, v4.2d, v14.2d
	fsub	v1.2d, v1.2d, v14.2d
	fsub	d22, d3, d21
	dup	v3.2d, v3.d[0]
	fmadd	d19, d19, d19, d20
	fmadd	d17, d18, d18, d17
	fmul	v5.2d, v5.2d, v5.2d
	fmul	v27.2d, v27.2d, v27.2d
	fsub	v3.2d, v3.2d, v15.2d
	fmadd	d19, d22, d22, d19
	ldur	q22, [x19, #248]
	fmla	v5.2d, v4.2d, v4.2d
	fmla	v27.2d, v1.2d, v1.2d
	fsqrt	d6, d19
	fsub	d19, d2, d21
	dup	v2.2d, v2.d[0]
	fmla	v5.2d, v3.2d, v3.2d
	fsub	v1.2d, v2.2d, v15.2d
	dup	v2.2d, v16.d[0]
	fmadd	d17, d19, d19, d17
	ldp	d19, d18, [x19, #224]
	fsub	v0.2d, v2.2d, v0.2d
	ldr	q2, [sp]                        // 16-byte Folded Reload
	fmla	v27.2d, v1.2d, v1.2d
	dup	v2.2d, v2.d[0]
	fmul	v0.2d, v0.2d, v0.2d
	fsub	v2.2d, v2.2d, v14.2d
	fmla	v0.2d, v2.2d, v2.2d
	fsqrt	d29, d17
	ldr	d17, [x19, #176]
	fsub	d17, d17, d18
	ldp	d26, d18, [x19, #160]
	fsub	d18, d18, d19
	ldr	d19, [x19, #184]
	fmadd	d28, d25, d26, d28
	fmul	d17, d17, d17
	fmul	d4, d10, d26
	fsub	d19, d19, d12
	fmadd	d17, d18, d18, d17
	ldur	q18, [x19, #24]
	fmla	v23.2d, v18.2d, v10.d[0]
	ldp	d20, d18, [x19, #264]
	fmadd	d17, d19, d19, d17
	ldp	d19, d21, [x19, #208]
	fmla	v23.2d, v9.2d, v8.d[0]
	fmadd	d28, d19, d21, d28
	fmla	v23.2d, v30.2d, v26.d[0]
	fmadd	d28, d20, d18, d28
	fmla	v23.2d, v24.2d, v21.d[0]
	fmla	v23.2d, v22.2d, v18.d[0]
	fsqrt	v3.2d, v5.2d
	mov	v5.16b, v21.16b
	mov	v5.d[1], v18.d[0]
	fdiv	v23.2d, v23.2d, v13.2d
	ldr	d13, [x8, :lo12:.LCPI3_0]
	fsqrt	v1.2d, v27.2d
	fmul	v2.2d, v23.2d, v23.2d
	stur	q23, [x19, #24]
	mov	d2, v2.d[1]
	fmadd	d2, d23, d23, d2
	fdiv	d28, d28, d13
	fmul	d13, d10, d8
	fdiv	d13, d13, d7
	ldr	d7, [x19, #128]
	dup	v7.2d, v7.d[0]
	fmadd	d2, d28, d28, d2
	str	d28, [x19, #40]
	fsub	v7.2d, v7.2d, v15.2d
	fmla	v0.2d, v7.2d, v7.2d
	fmul	v7.2d, v5.2d, v8.d[0]
	fdiv	d4, d4, d6
	fmul	v6.2d, v5.2d, v10.d[0]
	fmul	v5.2d, v5.2d, v26.d[0]
	fdiv	v3.2d, v6.2d, v3.2d
	fmul	d6, d8, d26
	fdiv	v1.2d, v7.2d, v1.2d
	fmov	d7, #0.50000000
	fmul	d16, d10, d7
	fmadd	d2, d16, d2, d11
	fmul	v16.2d, v9.2d, v9.2d
	fsub	d2, d2, d13
	mov	d16, v16.d[1]
	fsub	d2, d2, d4
	fmadd	d4, d9, d9, d16
	fsqrt	v0.2d, v0.2d
	fsub	d2, d2, d3
	mov	d3, v3.d[1]
	fmadd	d4, d31, d31, d4
	fsub	d2, d2, d3
	fmul	v3.2d, v30.2d, v30.2d
	mov	d3, v3.d[1]
	fmadd	d3, d30, d30, d3
	fmadd	d3, d25, d25, d3
	fdiv	d6, d6, d29
	fsqrt	d17, d17
	fdiv	v0.2d, v5.2d, v0.2d
	fmul	d5, d8, d7
	fmadd	d2, d5, d4, d2
	fmul	d4, d21, d18
	fmul	v5.2d, v24.2d, v24.2d
	fsub	d2, d2, d6
	fmul	d6, d26, d7
	fsub	d2, d2, d1
	mov	d1, v1.d[1]
	fsub	d1, d2, d1
	mov	d2, v5.d[1]
	fmul	d5, d21, d7
	fdiv	d4, d4, d17
	fmadd	d1, d6, d3, d1
	fmul	v3.2d, v22.2d, v22.2d
	fmadd	d2, d24, d24, d2
	fsub	d1, d1, d0
	mov	d0, v0.d[1]
	mov	d3, v3.d[1]
	fmadd	d2, d19, d19, d2
	fsub	d0, d1, d0
	fmadd	d1, d22, d22, d3
	fmadd	d0, d5, d2, d0
	fmul	d2, d18, d7
	fmadd	d1, d20, d20, d1
	fsub	d0, d0, d4
	fmadd	d0, d2, d1, d0
	bl	printf
	adrp	x8, .LCPI3_1
	mov	w20, #19264                     // =0x4b40
	ldr	d8, [x8, :lo12:.LCPI3_1]
	movk	w20, #76, lsl #16
.LBB3_1:                                // =>This Inner Loop Header: Depth=1
	fmov	d0, d8
	mov	w0, #5                          // =0x5
	mov	x1, x19
	bl	advance
	subs	w20, w20, #1
	b.ne	.LBB3_1
// %bb.2:
	ldp	d4, d5, [x19]
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	ldp	d2, d0, [x19, #56]
	ldr	d3, [x19, #16]
	ldr	d18, [x19, #112]
	ldur	q25, [x19, #168]
	ldr	q26, [x19, #224]
	fsub	d1, d5, d0
	fsub	d6, d4, d2
	fsub	d19, d4, d18
	fsub	d18, d2, d18
	zip2	v27.2d, v25.2d, v26.2d
	dup	v4.2d, v4.d[0]
	dup	v2.2d, v2.d[0]
	fmul	d7, d1, d1
	ldr	d1, [x19, #72]
	fsub	d16, d3, d1
	fmadd	d6, d6, d6, d7
	ldp	d7, d17, [x19, #112]
	fmadd	d6, d16, d16, d6
	fsub	d16, d5, d17
	fsub	d17, d0, d17
	dup	v5.2d, v5.d[0]
	dup	v0.2d, v0.d[0]
	fmul	d20, d16, d16
	ldp	d16, d21, [x19, #120]
	fmul	d17, d17, d17
	fsqrt	d6, d6
	fsub	v5.2d, v5.2d, v27.2d
	fsub	v0.2d, v0.2d, v27.2d
	fsub	d22, d3, d21
	dup	v3.2d, v3.d[0]
	fmadd	d19, d19, d19, d20
	fmadd	d17, d18, d18, d17
	fmul	v5.2d, v5.2d, v5.2d
	fmul	v0.2d, v0.2d, v0.2d
	fmadd	d19, d22, d22, d19
	fsqrt	d20, d19
	fsub	d19, d1, d21
	dup	v1.2d, v1.d[0]
	ldp	d21, d18, [x19, #224]
	fmadd	d17, d19, d19, d17
	fsqrt	d19, d17
	ldp	d17, d22, [x19, #176]
	fsub	d17, d17, d18
	ldr	d18, [x19, #168]
	fsub	d18, d18, d21
	ldr	d21, [x19, #240]
	fmul	d17, d17, d17
	fsub	d22, d22, d21
	fmadd	d17, d18, d18, d17
	ldp	d18, d23, [x19, #96]
	fmadd	d17, d22, d22, d17
	ldr	d22, [x19, #48]
	fmul	d24, d22, d23
	fdiv	d6, d24, d6
	zip1	v24.2d, v25.2d, v26.2d
	ldur	q25, [x19, #184]
	ldur	q26, [x19, #216]
	mov	v25.d[1], v21.d[0]
	mov	v28.16b, v26.16b
	mov	v26.d[1], v23.d[0]
	fsub	v4.2d, v4.2d, v24.2d
	fsub	v2.2d, v2.2d, v24.2d
	fsub	v3.2d, v3.2d, v25.2d
	fsub	v1.2d, v1.2d, v25.2d
	mov	v28.d[1], v22.d[0]
	fmla	v5.2d, v4.2d, v4.2d
	fmla	v0.2d, v2.2d, v2.2d
	ldr	d4, [x19, #272]
	dup	v2.2d, v7.d[0]
	ldr	d7, [x19, #128]
	dup	v7.2d, v7.d[0]
	fmla	v5.2d, v3.2d, v3.2d
	fmla	v0.2d, v1.2d, v1.2d
	dup	v1.2d, v16.d[0]
	fsub	v2.2d, v2.2d, v24.2d
	fsub	v7.2d, v7.2d, v25.2d
	fsub	v1.2d, v1.2d, v27.2d
	fsqrt	v21.2d, v5.2d
	ldr	d5, [x19, #160]
	fmul	v1.2d, v1.2d, v1.2d
	fmul	d3, d22, d5
	fmla	v1.2d, v2.2d, v2.2d
	fmla	v1.2d, v7.2d, v7.2d
	fdiv	d20, d3, d20
	fmov	d3, #0.50000000
	fmul	d29, d22, d3
	mov	v22.d[1], v4.d[0]
	fmul	d2, d23, d3
	fmul	v22.2d, v28.2d, v22.2d
	fsqrt	v0.2d, v0.2d
	fdiv	v21.2d, v22.2d, v21.2d
	fmul	d22, d23, d5
	mov	v23.d[1], v4.d[0]
	fmul	v16.2d, v26.2d, v23.2d
	fdiv	d19, d22, d19
	ldp	d22, d7, [x19, #24]
	fmul	d7, d7, d7
	fmadd	d7, d22, d22, d7
	ldr	d22, [x19, #216]
	fsqrt	v1.2d, v1.2d
	fdiv	v0.2d, v16.2d, v0.2d
	ldr	d16, [x19, #40]
	fmadd	d7, d16, d16, d7
	movi	d16, #0000000000000000
	fmadd	d7, d29, d7, d16
	mov	v16.16b, v22.16b
	mov	v16.d[1], v4.d[0]
	fsub	d6, d7, d6
	ldr	d7, [x19, #88]
	fmul	d7, d7, d7
	fmul	v16.2d, v16.2d, v5.d[0]
	fmul	d5, d5, d3
	fsub	d6, d6, d20
	ldr	d20, [x19, #80]
	fmadd	d7, d20, d20, d7
	fsub	d6, d6, d21
	fsqrt	d17, d17
	fmadd	d7, d18, d18, d7
	fdiv	v1.2d, v16.2d, v1.2d
	mov	d16, v21.d[1]
	fsub	d6, d6, d16
	fmul	d16, d22, d4
	fmadd	d2, d2, d7, d6
	ldp	d7, d6, [x19, #136]
	fmul	d6, d6, d6
	fsub	d2, d2, d19
	fmadd	d6, d7, d7, d6
	ldr	d7, [x19, #152]
	fsub	d2, d2, d0
	mov	d0, v0.d[1]
	fdiv	d16, d16, d17
	fmadd	d6, d7, d7, d6
	fmul	d7, d22, d3
	fsub	d0, d2, d0
	ldr	d2, [x19, #200]
	fmul	d3, d4, d3
	fmul	d2, d2, d2
	fmadd	d0, d5, d6, d0
	ldr	d5, [x19, #192]
	ldr	d6, [x19, #256]
	fmadd	d2, d5, d5, d2
	ldr	d5, [x19, #208]
	fmul	d6, d6, d6
	fsub	d0, d0, d1
	mov	d1, v1.d[1]
	fmadd	d2, d5, d5, d2
	fsub	d0, d0, d1
	ldr	d1, [x19, #248]
	fmadd	d1, d1, d1, d6
	fmadd	d0, d7, d2, d0
	ldr	d2, [x19, #264]
	fmadd	d1, d2, d2, d1
	fsub	d0, d0, d16
	fmadd	d0, d3, d1, d0
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x29, x30, [sp, #96]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #80]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #64]             // 16-byte Folded Reload
	ldp	d13, d12, [sp, #48]             // 16-byte Folded Reload
	ldp	d15, d14, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #128
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
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
