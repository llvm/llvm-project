	.file	"puzzle.c"
	.text
	.globl	rand                            // -- Begin function rand
	.p2align	2
	.type	rand,@function
rand:                                   // @rand
	.cfi_startproc
// %bb.0:
	adrp	x0, next
	mov	w2, #20077                      // =0x4e6d
	mov	w3, #12345                      // =0x3039
	ldr	x1, [x0, :lo12:next]
	movk	w2, #16838, lsl #16
	madd	x1, x1, x2, x3
	mov	w3, #5                          // =0x5
	movk	w3, #2, lsl #16
	ubfx	x2, x1, #16, #32
	umull	x3, w2, w3
	lsr	x2, x1, #16
	lsr	x3, x3, #32
	sub	w4, w2, w3
	add	w3, w3, w4, lsr #1
	lsr	w3, w3, #14
	sub	w3, w3, w3, lsl #15
	add	w2, w2, w3
	add	w0, w2, #1
	str	x1, [x0, :lo12:next]
	ret
.Lfunc_end0:
	.size	rand, .Lfunc_end0-rand
	.cfi_endproc
                                        // -- End function
	.globl	srand                           // -- Begin function srand
	.p2align	2
	.type	srand,@function
srand:                                  // @srand
	.cfi_startproc
// %bb.0:
	mov	w1, w0
	adrp	x0, next
	mov	w1, w1
	str	x1, [x0, :lo12:next]
	ret
.Lfunc_end1:
	.size	srand, .Lfunc_end1-srand
	.cfi_endproc
                                        // -- End function
	.globl	randInt                         // -- Begin function randInt
	.p2align	2
	.type	randInt,@function
randInt:                                // @randInt
	.cfi_startproc
// %bb.0:
	mov	w2, w1
	adrp	x1, next
	mov	w4, #20077                      // =0x4e6d
	ldr	x3, [x1, :lo12:next]
	movk	w4, #16838, lsl #16
	mov	w5, #12345                      // =0x3039
	sub	w2, w2, w0
	madd	x3, x3, x4, x5
	mov	w5, #5                          // =0x5
	add	w2, w2, #1
	movk	w5, #2, lsl #16
	scvtf	d0, w2
	ubfx	x4, x3, #16, #32
	str	x3, [x1, :lo12:next]
	umull	x5, w4, w5
	lsr	x4, x3, #16
	lsr	x5, x5, #32
	sub	w6, w4, w5
	add	w5, w5, w6, lsr #1
	lsr	w5, w5, #14
	sub	w5, w5, w5, lsl #15
	add	w4, w4, w5
	add	w4, w4, #1
	ucvtf	d1, w4, #15
	fmul	d0, d1, d0
	fcvtzs	w4, d0
	cmp	w2, w4
	add	w0, w0, w4
	cset	w2, eq
	sub	w0, w0, w2
	ret
.Lfunc_end2:
	.size	randInt, .Lfunc_end2-randInt
	.cfi_endproc
                                        // -- End function
	.globl	shuffle                         // -- Begin function shuffle
	.p2align	2
	.type	shuffle,@function
shuffle:                                // @shuffle
	.cfi_startproc
// %bb.0:
	cmp	w1, #1
	b.eq	.LBB3_4
// %bb.1:
	mov	w3, w1
	adrp	x1, next
	mov	w5, #20077                      // =0x4e6d
	ldr	x2, [x1, :lo12:next]
	sxtw	x3, w3
	mov	w7, #5                          // =0x5
	sub	x4, x0, #4
	movk	w5, #16838, lsl #16
	mov	w6, #12345                      // =0x3039
	movk	w7, #2, lsl #16
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	madd	x2, x2, x5, x6
	ucvtf	d0, x3
	ubfx	x8, x2, #16, #32
	umull	x9, w8, w7
	lsr	x8, x2, #16
	lsr	x9, x9, #32
	sub	w10, w8, w9
	add	w9, w9, w10, lsr #1
	ldr	w10, [x4, x3, lsl #2]
	lsr	w9, w9, #14
	sub	w9, w9, w9, lsl #15
	add	w8, w8, w9
	add	w8, w8, #1
	ucvtf	d1, w8, #15
	fmul	d0, d1, d0
	fcvtzs	w8, d0
	cmp	x3, w8, sxtw
	add	x8, x0, w8, sxtw #2
	csetm	x9, eq
	ldr	w11, [x8, x9, lsl #2]
	str	w11, [x4, x3, lsl #2]
	sub	x3, x3, #1
	cmp	x3, #1
	str	w10, [x8, x9, lsl #2]
	b.ne	.LBB3_2
// %bb.3:
	str	x2, [x1, :lo12:next]
.LBB3_4:
	ret
.Lfunc_end3:
	.size	shuffle, .Lfunc_end3-shuffle
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function createRandomArray
.LCPI4_0:
	.word	0                               // 0x0
	.word	1                               // 0x1
	.word	2                               // 0x2
	.word	3                               // 0x3
	.text
	.globl	createRandomArray
	.p2align	2
	.type	createRandomArray,@function
createRandomArray:                      // @createRandomArray
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	add	w2, w0, #1
	sbfiz	x0, x2, #2, #32
	bl	malloc
	tbnz	w1, #31, .LBB4_7
// %bb.1:
	cmp	w2, #8
	b.hs	.LBB4_3
// %bb.2:
	mov	x3, xzr
	b	.LBB4_6
.LBB4_3:
	movi	v0.4s, #4
	movi	v1.4s, #8
	adrp	x4, .LCPI4_0
	and	x3, x2, #0xfffffff8
	ldr	q2, [x4, :lo12:.LCPI4_0]
	add	x4, x0, #16
	mov	x5, x3
.LBB4_4:                                // =>This Inner Loop Header: Depth=1
	add	v3.4s, v2.4s, v0.4s
	subs	x5, x5, #8
	stp	q2, q3, [x4, #-16]
	add	v2.4s, v2.4s, v1.4s
	add	x4, x4, #32
	b.ne	.LBB4_4
// %bb.5:
	cmp	x3, x2
	b.eq	.LBB4_7
.LBB4_6:                                // =>This Inner Loop Header: Depth=1
	str	w3, [x0, x3, lsl #2]
	add	x3, x3, #1
	cmp	x2, x3
	b.ne	.LBB4_6
.LBB4_7:
	adrp	x2, next
	mov	w3, #20077                      // =0x4e6d
	mov	w5, #12345                      // =0x3039
	ldr	x4, [x2, :lo12:next]
	movk	w3, #16838, lsl #16
	mov	w6, #5                          // =0x5
	movk	w6, #2, lsl #16
	scvtf	d0, w1
	madd	x4, x4, x3, x5
	ubfx	x5, x4, #16, #32
	str	x4, [x2, :lo12:next]
	umull	x6, w5, w6
	lsr	x5, x4, #16
	lsr	x6, x6, #32
	sub	w7, w5, w6
	add	w6, w6, w7, lsr #1
	lsr	w6, w6, #14
	sub	w6, w6, w6, lsl #15
	add	w5, w5, w6
	add	w5, w5, #1
	ucvtf	d1, w5, #15
	fmul	d0, d1, d0
	fcvtzs	w5, d0
	cmp	w1, w5
	cset	w6, eq
	sub	w5, w5, w6
	add	w5, w5, #1
	str	w5, [x0]
	cbz	w1, .LBB4_11
// %bb.8:
	sxtw	x1, w1
	mov	w6, #5                          // =0x5
	mov	w5, #12345                      // =0x3039
	movk	w6, #2, lsl #16
.LBB4_9:                                // =>This Inner Loop Header: Depth=1
	madd	x4, x4, x3, x5
	ubfx	x7, x4, #16, #32
	umull	x8, w7, w6
	lsr	x7, x4, #16
	lsr	x8, x8, #32
	sub	w9, w7, w8
	add	w8, w8, w9, lsr #1
	ldr	w9, [x0, x1, lsl #2]
	lsr	w8, w8, #14
	sub	w8, w8, w8, lsl #15
	add	w8, w7, w8
	add	x7, x1, #1
	add	w8, w8, #1
	ucvtf	d0, x7
	ucvtf	d1, w8, #15
	fmul	d0, d1, d0
	fcvtzs	w8, d0
	cmp	x7, w8, sxtw
	add	x7, x0, w8, sxtw #2
	csetm	x8, eq
	ldr	w10, [x7, x8, lsl #2]
	str	w10, [x0, x1, lsl #2]
	subs	x1, x1, #1
	str	w9, [x7, x8, lsl #2]
	b.ne	.LBB4_9
// %bb.10:
	str	x4, [x2, :lo12:next]
.LBB4_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end4:
	.size	createRandomArray, .Lfunc_end4-createRandomArray
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function findDuplicate
.LCPI5_0:
	.xword	2                               // 0x2
	.xword	3                               // 0x3
.LCPI5_1:
	.xword	0                               // 0x0
	.xword	1                               // 0x1
	.text
	.globl	findDuplicate
	.p2align	2
	.type	findDuplicate,@function
findDuplicate:                          // @findDuplicate
	.cfi_startproc
// %bb.0:
	mov	w0, w1
	cmp	w1, #1
	b.lt	.LBB5_3
// %bb.1:
	stp	d9, d8, [sp, #-16]!             // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_remember_state
	mov	x1, x0
	cmp	w0, #8
	mov	w2, w0
	b.hs	.LBB5_4
// %bb.2:
	mov	x3, xzr
	mov	w4, wzr
	b	.LBB5_7
.LBB5_3:
	.cfi_def_cfa wsp, 0
	.cfi_same_value b8
	.cfi_same_value b9
	eor	w0, wzr, w0
	ret
.LBB5_4:
	.cfi_restore_state
	movi	v0.2d, #0000000000000000
	movi	v1.4s, #1
	mov	w3, #8                          // =0x8
	movi	v2.4s, #5
	movi	v3.2d, #0000000000000000
	adrp	x5, .LCPI5_0
	adrp	x4, .LCPI5_1
	dup	v4.2d, x3
	and	x3, x2, #0x7ffffff8
	ldr	q5, [x5, :lo12:.LCPI5_0]
	ldr	q6, [x4, :lo12:.LCPI5_1]
	add	x4, x1, #16
	mov	x5, x3
.LBB5_5:                                // =>This Inner Loop Header: Depth=1
	uzp1	v7.4s, v6.4s, v5.4s
	ldp	q8, q9, [x4, #-16]
	add	v5.2d, v5.2d, v4.2d
	add	v6.2d, v6.2d, v4.2d
	subs	x5, x5, #8
	add	x4, x4, #32
	eor	v0.16b, v0.16b, v8.16b
	eor	v3.16b, v3.16b, v9.16b
	add	v8.4s, v7.4s, v1.4s
	add	v7.4s, v7.4s, v2.4s
	eor	v0.16b, v0.16b, v8.16b
	eor	v3.16b, v3.16b, v7.16b
	b.ne	.LBB5_5
// %bb.6:
	eor	v0.16b, v3.16b, v0.16b
	cmp	x3, x2
	ext	v1.16b, v0.16b, v0.16b, #8
	eor	v0.8b, v0.8b, v1.8b
	fmov	x4, d0
	lsr	x5, x4, #32
	eor	w4, w4, w5
	b.eq	.LBB5_9
.LBB5_7:                                // %.preheader
	add	x1, x1, x3, lsl #2
	sub	x2, x2, x3
	add	w3, w3, #1
.LBB5_8:                                // =>This Inner Loop Header: Depth=1
	ldr	w5, [x1], #4
	eor	w4, w4, w3
	subs	x2, x2, #1
	add	w3, w3, #1
	eor	w4, w4, w5
	b.ne	.LBB5_8
.LBB5_9:
	ldp	d9, d8, [sp], #16               // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	eor	w0, w4, w0
	ret
.Lfunc_end5:
	.size	findDuplicate, .Lfunc_end5-findDuplicate
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function main
.LCPI6_0:
	.word	0                               // 0x0
	.word	1                               // 0x1
	.word	2                               // 0x2
	.word	3                               // 0x3
.LCPI6_1:
	.xword	2                               // 0x2
	.xword	3                               // 0x3
.LCPI6_2:
	.xword	0                               // 0x0
	.xword	1                               // 0x1
	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	stp	d15, d14, [sp, #-80]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	d13, d12, [sp, #16]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #64
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
	adrp	x7, .LCPI6_2
	adrp	x2, next
	mov	w4, #1                          // =0x1
	ldr	q7, [x7, :lo12:.LCPI6_2]
	mov	x7, #145685290680320            // =0x848000000000
	movi	v1.4s, #4
	movi	v2.4s, #8
	movi	v3.4s, #1
	movi	v4.4s, #5
	mov	w8, #8                          // =0x8
	movk	x7, #16670, lsl #48
	str	x4, [x2, :lo12:next]
	adrp	x4, .LCPI6_0
	adrp	x6, .LCPI6_1
	dup	v6.2d, x8
	fmov	d8, x7
	mov	w1, #20077                      // =0x4e6d
	mov	w3, #41248                      // =0xa120
	ldr	q0, [x4, :lo12:.LCPI6_0]
	mov	w4, #33920                      // =0x8480
	ldr	q5, [x6, :lo12:.LCPI6_1]
	mov	w6, #5                          // =0x5
	mov	w0, wzr
	movk	w1, #16838, lsl #16
	movk	w3, #7, lsl #16
	movk	w4, #30, lsl #16
	mov	w5, #12345                      // =0x3039
	movk	w6, #2, lsl #16
	adrp	x7, .L.str
	add	x7, x7, :lo12:.L.str
.LBB6_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_2 Depth 2
                                        //     Child Loop BB6_4 Depth 2
                                        //     Child Loop BB6_6 Depth 2
                                        //       Child Loop BB6_7 Depth 3
	mov	w0, #33924                      // =0x8484
	movk	w0, #30, lsl #16
	bl	malloc
	mov	v9.16b, v0.16b
	mov	w10, #41248                     // =0xa120
	add	x9, x0, #16
	mov	x8, x0
	movk	w10, #7, lsl #16
	mov	x11, x9
.LBB6_2:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	v10.4s, v9.4s, v1.4s
	subs	x10, x10, #8
	stp	q9, q10, [x11, #-16]
	add	v9.4s, v9.4s, v2.4s
	add	x11, x11, #32
	b.ne	.LBB6_2
// %bb.3:                               //   in Loop: Header=BB6_1 Depth=1
	ldr	x10, [x2, :lo12:next]
	str	w3, [x8, x4]
	madd	x10, x10, x1, x5
	ubfx	x11, x10, #16, #32
	umull	x12, w11, w6
	lsr	x11, x10, #16
	lsr	x12, x12, #32
	sub	w13, w11, w12
	add	w12, w12, w13, lsr #1
	lsr	w12, w12, #14
	sub	w12, w12, w12, lsl #15
	add	w11, w11, w12
	add	w11, w11, #1
	ucvtf	d9, w11, #15
	fmul	d9, d9, d8
	fcvtzs	w11, d9
	cmp	w11, w3
	cset	w12, eq
	sub	w11, w11, w12
	add	w11, w11, #1
	str	w11, [x8]
	mov	w11, #41248                     // =0xa120
	movk	w11, #7, lsl #16
.LBB6_4:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	madd	x10, x10, x1, x5
	ubfx	x12, x10, #16, #32
	umull	x13, w12, w6
	lsr	x12, x10, #16
	lsr	x13, x13, #32
	sub	w14, w12, w13
	add	w13, w13, w14, lsr #1
	ldr	w14, [x8, x11, lsl #2]
	lsr	w13, w13, #14
	sub	w13, w13, w13, lsl #15
	add	w13, w12, w13
	add	x12, x11, #1
	add	w13, w13, #1
	ucvtf	d9, x12
	ucvtf	d10, w13, #15
	fmul	d9, d10, d9
	fcvtzs	w13, d9
	cmp	x12, w13, sxtw
	add	x12, x8, w13, sxtw #2
	csetm	x13, eq
	ldr	w15, [x12, x13, lsl #2]
	str	w15, [x8, x11, lsl #2]
	subs	x11, x11, #1
	str	w14, [x12, x13, lsl #2]
	b.ne	.LBB6_4
// %bb.5:                               //   in Loop: Header=BB6_1 Depth=1
	str	x10, [x2, :lo12:next]
.LBB6_6:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_7 Depth 3
	movi	v9.2d, #0000000000000000
	movi	v10.2d, #0000000000000000
	mov	w12, #41248                     // =0xa120
	mov	v11.16b, v7.16b
	mov	v12.16b, v5.16b
	mov	x10, x9
	movk	w12, #7, lsl #16
.LBB6_7:                                //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	uzp1	v13.4s, v11.4s, v12.4s
	ldp	q15, q14, [x10, #-16]
	add	v12.2d, v12.2d, v6.2d
	add	v11.2d, v11.2d, v6.2d
	subs	x12, x12, #8
	add	x10, x10, #32
	eor	v9.16b, v15.16b, v9.16b
	eor	v10.16b, v14.16b, v10.16b
	add	v15.4s, v13.4s, v3.4s
	add	v13.4s, v13.4s, v4.4s
	eor	v9.16b, v9.16b, v15.16b
	eor	v10.16b, v10.16b, v13.16b
	b.ne	.LBB6_7
// %bb.8:                               //   in Loop: Header=BB6_6 Depth=2
	add	w11, w11, #1
	cmp	w11, #200
	b.ne	.LBB6_6
// %bb.9:                               //   in Loop: Header=BB6_1 Depth=1
	eor	v9.16b, v10.16b, v9.16b
	ldr	w10, [x8, x4]
	mov	x0, x8
	ext	v10.16b, v9.16b, v9.16b, #8
	eor	v9.8b, v9.8b, v10.8b
	fmov	x11, d9
	lsr	x9, x11, #32
	eor	w10, w10, w11
	bl	free
	eor	w1, w10, w9
	mov	x0, x7
	bl	printf
	add	w0, w0, #1
	cmp	w0, #5
	b.ne	.LBB6_1
// %bb.10:
	mov	w0, wzr
	.cfi_def_cfa wsp, 80
	ldp	x29, x30, [sp, #64]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             // 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             // 16-byte Folded Reload
	ldp	d15, d14, [sp], #80             // 16-byte Folded Reload
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
.Lfunc_end6:
	.size	main, .Lfunc_end6-main
	.cfi_endproc
                                        // -- End function
	.type	next,@object                    // @next
	.data
	.p2align	3, 0x0
next:
	.xword	1                               // 0x1
	.size	next, 8

	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Found duplicate: %d\n"
	.size	.L.str, 21

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
