	.file	"puzzle.c"
	.text
	.globl	rand                            // -- Begin function rand
	.p2align	2
	.type	rand,@function
rand:                                   // @rand
	.cfi_startproc
// %bb.0:
	adrp	x8, next
	mov	w10, #20077                     // =0x4e6d
	mov	w11, #12345                     // =0x3039
	ldr	x9, [x8, :lo12:next]
	movk	w10, #16838, lsl #16
	madd	x9, x9, x10, x11
	mov	w11, #5                         // =0x5
	movk	w11, #2, lsl #16
	ubfx	x10, x9, #16, #32
	str	x9, [x8, :lo12:next]
	umull	x10, w10, w11
	lsr	x11, x9, #16
	lsr	x10, x10, #32
	sub	w12, w11, w10
	add	w10, w10, w12, lsr #1
	lsr	w10, w10, #14
	sub	w10, w10, w10, lsl #15
	add	w10, w11, w10
	add	w0, w10, #1
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
	adrp	x8, next
	mov	w9, w0
	str	x9, [x8, :lo12:next]
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
	adrp	x8, next
	mov	w10, #20077                     // =0x4e6d
	mov	w11, #12345                     // =0x3039
	ldr	x9, [x8, :lo12:next]
	movk	w10, #16838, lsl #16
	madd	x9, x9, x10, x11
	mov	w11, #5                         // =0x5
	movk	w11, #2, lsl #16
	ubfx	x10, x9, #16, #32
	str	x9, [x8, :lo12:next]
	umull	x10, w10, w11
	lsr	x11, x9, #16
	lsr	x10, x10, #32
	sub	w12, w11, w10
	add	w10, w10, w12, lsr #1
	sub	w12, w1, w0
	lsr	w10, w10, #14
	sub	w10, w10, w10, lsl #15
	add	w10, w11, w10
	add	w11, w12, #1
	add	w10, w10, #1
	scvtf	d0, w11
	ucvtf	d1, w10, #15
	fmul	d0, d1, d0
	fcvtzs	w10, d0
	cmp	w11, w10
	add	w10, w0, w10
	cset	w11, eq
	sub	w0, w10, w11
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
                                        // kill: def $w1 killed $w1 def $x1
	cmp	w1, #1
	b.eq	.LBB3_4
// %bb.1:
	adrp	x8, next
	sxtw	x10, w1
	mov	w12, #20077                     // =0x4e6d
	ldr	x9, [x8, :lo12:next]
	mov	w14, #5                         // =0x5
	sub	x11, x0, #4
	movk	w12, #16838, lsl #16
	mov	w13, #12345                     // =0x3039
	movk	w14, #2, lsl #16
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	madd	x9, x9, x12, x13
	ucvtf	d0, x10
	ldr	w18, [x11, x10, lsl #2]
	ubfx	x15, x9, #16, #32
	lsr	x16, x9, #16
	umull	x15, w15, w14
	lsr	x15, x15, #32
	sub	w17, w16, w15
	add	w15, w15, w17, lsr #1
	lsr	w15, w15, #14
	sub	w15, w15, w15, lsl #15
	add	w15, w16, w15
	add	w15, w15, #1
	ucvtf	d1, w15, #15
	fmul	d0, d1, d0
	fcvtzs	w15, d0
	cmp	x10, w15, sxtw
	add	x15, x0, w15, sxtw #2
	csetm	x16, eq
	ldr	w17, [x15, x16, lsl #2]
	str	w17, [x11, x10, lsl #2]
	sub	x10, x10, #1
	cmp	x10, #1
	str	w18, [x15, x16, lsl #2]
	b.ne	.LBB3_2
// %bb.3:
	str	x9, [x8, :lo12:next]
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
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x20, x19, [sp, #16]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	add	w20, w0, #1
	mov	w19, w0
	sbfiz	x0, x20, #2, #32
	bl	malloc
	tbnz	w19, #31, .LBB4_7
// %bb.1:
	cmp	w20, #8
	b.hs	.LBB4_3
// %bb.2:
	mov	x8, xzr
	b	.LBB4_6
.LBB4_3:
	movi	v0.4s, #4
	movi	v1.4s, #8
	adrp	x9, .LCPI4_0
	and	x8, x20, #0xfffffff8
	ldr	q2, [x9, :lo12:.LCPI4_0]
	add	x9, x0, #16
	mov	x10, x8
.LBB4_4:                                // =>This Inner Loop Header: Depth=1
	add	v3.4s, v2.4s, v0.4s
	subs	x10, x10, #8
	stp	q2, q3, [x9, #-16]
	add	v2.4s, v2.4s, v1.4s
	add	x9, x9, #32
	b.ne	.LBB4_4
// %bb.5:
	cmp	x8, x20
	b.eq	.LBB4_7
.LBB4_6:                                // =>This Inner Loop Header: Depth=1
	str	w8, [x0, x8, lsl #2]
	add	x8, x8, #1
	cmp	x20, x8
	b.ne	.LBB4_6
.LBB4_7:
	adrp	x8, next
	mov	w9, #20077                      // =0x4e6d
	mov	w11, #12345                     // =0x3039
	ldr	x10, [x8, :lo12:next]
	movk	w9, #16838, lsl #16
	mov	w12, #5                         // =0x5
	movk	w12, #2, lsl #16
	scvtf	d0, w19
	madd	x10, x10, x9, x11
	ubfx	x11, x10, #16, #32
	str	x10, [x8, :lo12:next]
	umull	x11, w11, w12
	lsr	x12, x10, #16
	lsr	x11, x11, #32
	sub	w13, w12, w11
	add	w11, w11, w13, lsr #1
	lsr	w11, w11, #14
	sub	w11, w11, w11, lsl #15
	add	w11, w12, w11
	add	w11, w11, #1
	ucvtf	d1, w11, #15
	fmul	d0, d1, d0
	fcvtzs	w11, d0
	cmp	w19, w11
	cset	w12, eq
	sub	w11, w11, w12
	add	w11, w11, #1
	str	w11, [x0]
	cbz	w19, .LBB4_11
// %bb.8:
	sxtw	x11, w19
	mov	w13, #5                         // =0x5
	mov	w12, #12345                     // =0x3039
	movk	w13, #2, lsl #16
.LBB4_9:                                // =>This Inner Loop Header: Depth=1
	madd	x10, x10, x9, x12
	ldr	w17, [x0, x11, lsl #2]
	ubfx	x14, x10, #16, #32
	lsr	x15, x10, #16
	umull	x14, w14, w13
	lsr	x14, x14, #32
	sub	w16, w15, w14
	add	w14, w14, w16, lsr #1
	lsr	w14, w14, #14
	sub	w14, w14, w14, lsl #15
	add	w14, w15, w14
	add	x15, x11, #1
	add	w14, w14, #1
	ucvtf	d0, x15
	ucvtf	d1, w14, #15
	fmul	d0, d1, d0
	fcvtzs	w14, d0
	cmp	x15, w14, sxtw
	add	x14, x0, w14, sxtw #2
	csetm	x15, eq
	ldr	w16, [x14, x15, lsl #2]
	str	w16, [x0, x11, lsl #2]
	subs	x11, x11, #1
	str	w17, [x14, x15, lsl #2]
	b.ne	.LBB4_9
// %bb.10:
	str	x10, [x8, :lo12:next]
.LBB4_11:
	.cfi_def_cfa wsp, 32
	ldp	x20, x19, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
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
	cmp	w1, #1
	b.lt	.LBB5_3
// %bb.1:
	cmp	w1, #8
	mov	w8, w1
	b.hs	.LBB5_4
// %bb.2:
	mov	x9, xzr
	mov	w10, wzr
	b	.LBB5_7
.LBB5_3:
	eor	w0, wzr, w1
	ret
.LBB5_4:
	movi	v1.2d, #0000000000000000
	movi	v0.4s, #1
	mov	w9, #8                          // =0x8
	movi	v2.4s, #5
	movi	v3.2d, #0000000000000000
	adrp	x10, .LCPI5_0
	adrp	x11, .LCPI5_1
	dup	v4.2d, x9
	and	x9, x8, #0x7ffffff8
	ldr	q5, [x10, :lo12:.LCPI5_0]
	ldr	q6, [x11, :lo12:.LCPI5_1]
	add	x10, x0, #16
	mov	x11, x9
.LBB5_5:                                // =>This Inner Loop Header: Depth=1
	uzp1	v7.4s, v6.4s, v5.4s
	ldp	q16, q17, [x10, #-16]
	add	v5.2d, v5.2d, v4.2d
	add	v6.2d, v6.2d, v4.2d
	subs	x11, x11, #8
	add	x10, x10, #32
	eor	v1.16b, v1.16b, v16.16b
	eor	v3.16b, v3.16b, v17.16b
	add	v16.4s, v7.4s, v0.4s
	add	v7.4s, v7.4s, v2.4s
	eor	v1.16b, v1.16b, v16.16b
	eor	v3.16b, v3.16b, v7.16b
	b.ne	.LBB5_5
// %bb.6:
	eor	v0.16b, v3.16b, v1.16b
	cmp	x9, x8
	ext	v1.16b, v0.16b, v0.16b, #8
	eor	v0.8b, v0.8b, v1.8b
	fmov	x10, d0
	lsr	x11, x10, #32
	eor	w10, w10, w11
	b.eq	.LBB5_9
.LBB5_7:                                // %.preheader
	add	x11, x0, x9, lsl #2
	sub	x8, x8, x9
	add	w9, w9, #1
.LBB5_8:                                // =>This Inner Loop Header: Depth=1
	ldr	w12, [x11], #4
	eor	w10, w10, w9
	subs	x8, x8, #1
	add	w9, w9, #1
	eor	w10, w10, w12
	b.ne	.LBB5_8
.LBB5_9:
	eor	w0, w10, w1
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
	sub	sp, sp, #176
	.cfi_def_cfa_offset 176
	str	d8, [sp, #64]                   // 8-byte Folded Spill
	stp	x29, x30, [sp, #80]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #96]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #112]            // 16-byte Folded Spill
	stp	x24, x23, [sp, #128]            // 16-byte Folded Spill
	stp	x22, x21, [sp, #144]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #160]            // 16-byte Folded Spill
	add	x29, sp, #80
	.cfi_def_cfa w29, 96
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w27, -72
	.cfi_offset w28, -80
	.cfi_offset w30, -88
	.cfi_offset w29, -96
	.cfi_offset b8, -112
	adrp	x22, next
	mov	w8, #1                          // =0x1
	mov	w9, #8                          // =0x8
	str	x8, [x22, :lo12:next]
	adrp	x8, .LCPI6_0
	mov	w21, #20077                     // =0x4e6d
	ldr	q0, [x8, :lo12:.LCPI6_0]
	adrp	x8, .LCPI6_1
	mov	w23, #41248                     // =0xa120
	ldr	q1, [x8, :lo12:.LCPI6_1]
	adrp	x8, .LCPI6_2
	mov	w24, #33920                     // =0x8480
	stur	q0, [x29, #-32]                 // 16-byte Folded Spill
	dup	v0.2d, x9
	mov	w26, #5                         // =0x5
	mov	w20, wzr
	movk	w21, #16838, lsl #16
	movk	w23, #7, lsl #16
	movk	w24, #30, lsl #16
	mov	w25, #12345                     // =0x3039
	movk	w26, #2, lsl #16
	stp	q0, q1, [sp, #16]               // 32-byte Folded Spill
	ldr	q0, [x8, :lo12:.LCPI6_2]
	mov	x8, #145685290680320            // =0x848000000000
	movk	x8, #16670, lsl #48
	adrp	x19, .L.str
	add	x19, x19, :lo12:.L.str
	fmov	d8, x8
	str	q0, [sp]                        // 16-byte Folded Spill
.LBB6_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_2 Depth 2
                                        //     Child Loop BB6_4 Depth 2
                                        //     Child Loop BB6_6 Depth 2
                                        //       Child Loop BB6_7 Depth 3
	mov	w0, #33924                      // =0x8484
	movk	w0, #30, lsl #16
	bl	malloc
	movi	v2.4s, #4
	movi	v3.4s, #8
	mov	w9, #41248                      // =0xa120
	add	x8, x0, #16
	ldur	q0, [x29, #-32]                 // 16-byte Folded Reload
	movk	w9, #7, lsl #16
	mov	x10, x8
.LBB6_2:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	v1.4s, v0.4s, v2.4s
	subs	x9, x9, #8
	stp	q0, q1, [x10, #-16]
	add	v0.4s, v0.4s, v3.4s
	add	x10, x10, #32
	b.ne	.LBB6_2
// %bb.3:                               //   in Loop: Header=BB6_1 Depth=1
	ldr	x9, [x22, :lo12:next]
	str	w23, [x0, x24]
	madd	x9, x9, x21, x25
	ubfx	x10, x9, #16, #32
	lsr	x11, x9, #16
	umull	x10, w10, w26
	lsr	x10, x10, #32
	sub	w12, w11, w10
	add	w10, w10, w12, lsr #1
	lsr	w10, w10, #14
	sub	w10, w10, w10, lsl #15
	add	w10, w11, w10
	add	w10, w10, #1
	ucvtf	d0, w10, #15
	fmul	d0, d0, d8
	fcvtzs	w10, d0
	cmp	w10, w23
	cset	w11, eq
	sub	w10, w10, w11
	add	w10, w10, #1
	str	w10, [x0]
	mov	w10, #41248                     // =0xa120
	movk	w10, #7, lsl #16
.LBB6_4:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	madd	x9, x9, x21, x25
	ldr	w14, [x0, x10, lsl #2]
	ubfx	x11, x9, #16, #32
	lsr	x12, x9, #16
	umull	x11, w11, w26
	lsr	x11, x11, #32
	sub	w13, w12, w11
	add	w11, w11, w13, lsr #1
	lsr	w11, w11, #14
	sub	w11, w11, w11, lsl #15
	add	w11, w12, w11
	add	x12, x10, #1
	add	w11, w11, #1
	ucvtf	d0, x12
	ucvtf	d1, w11, #15
	fmul	d0, d1, d0
	fcvtzs	w11, d0
	cmp	x12, w11, sxtw
	add	x11, x0, w11, sxtw #2
	csetm	x12, eq
	ldr	w13, [x11, x12, lsl #2]
	str	w13, [x0, x10, lsl #2]
	subs	x10, x10, #1
	str	w14, [x11, x12, lsl #2]
	b.ne	.LBB6_4
// %bb.5:                               //   in Loop: Header=BB6_1 Depth=1
	movi	v7.4s, #1
	movi	v16.4s, #5
	ldr	q19, [sp]                       // 16-byte Folded Reload
	ldp	q18, q17, [sp, #16]             // 32-byte Folded Reload
	str	x9, [x22, :lo12:next]
.LBB6_6:                                //   Parent Loop BB6_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_7 Depth 3
	movi	v0.2d, #0000000000000000
	movi	v1.2d, #0000000000000000
	mov	w11, #41248                     // =0xa120
	mov	v2.16b, v19.16b
	mov	v3.16b, v17.16b
	mov	x9, x8
	movk	w11, #7, lsl #16
.LBB6_7:                                //   Parent Loop BB6_1 Depth=1
                                        //     Parent Loop BB6_6 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	uzp1	v4.4s, v2.4s, v3.4s
	ldp	q5, q6, [x9, #-16]
	add	v3.2d, v3.2d, v18.2d
	add	v2.2d, v2.2d, v18.2d
	subs	x11, x11, #8
	add	x9, x9, #32
	eor	v0.16b, v5.16b, v0.16b
	eor	v1.16b, v6.16b, v1.16b
	add	v5.4s, v4.4s, v7.4s
	add	v4.4s, v4.4s, v16.4s
	eor	v0.16b, v0.16b, v5.16b
	eor	v1.16b, v1.16b, v4.16b
	b.ne	.LBB6_7
// %bb.8:                               //   in Loop: Header=BB6_6 Depth=2
	add	w10, w10, #1
	cmp	w10, #200
	b.ne	.LBB6_6
// %bb.9:                               //   in Loop: Header=BB6_1 Depth=1
	eor	v0.16b, v1.16b, v0.16b
	ldr	w8, [x0, x24]
	ext	v1.16b, v0.16b, v0.16b, #8
	eor	v0.8b, v0.8b, v1.8b
	fmov	x9, d0
	lsr	x27, x9, #32
	eor	w28, w8, w9
	bl	free
	eor	w1, w28, w27
	mov	x0, x19
	bl	printf
	add	w20, w20, #1
	cmp	w20, #5
	b.ne	.LBB6_1
// %bb.10:
	mov	w0, wzr
	.cfi_def_cfa wsp, 176
	ldp	x20, x19, [sp, #160]            // 16-byte Folded Reload
	ldr	d8, [sp, #64]                   // 8-byte Folded Reload
	ldp	x22, x21, [sp, #144]            // 16-byte Folded Reload
	ldp	x24, x23, [sp, #128]            // 16-byte Folded Reload
	ldp	x26, x25, [sp, #112]            // 16-byte Folded Reload
	ldp	x28, x27, [sp, #96]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #80]             // 16-byte Folded Reload
	add	sp, sp, #176
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w27
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
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
