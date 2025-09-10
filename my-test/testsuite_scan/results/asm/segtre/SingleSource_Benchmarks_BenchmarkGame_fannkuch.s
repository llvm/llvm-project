	.file	"fannkuch.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function main
.LCPI0_0:
	.word	1                               // 0x1
	.word	2                               // 0x2
	.word	3                               // 0x3
	.word	4                               // 0x4
.LCPI0_1:
	.word	5                               // 0x5
	.word	6                               // 0x6
	.word	7                               // 0x7
	.word	8                               // 0x8
.LCPI0_3:
	.xword	0                               // 0x0
	.xword	1                               // 0x1
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0
.LCPI0_2:
	.word	9                               // 0x9
	.word	10                              // 0xa
	.text
	.globl	main
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
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	mov	w11, #11                        // =0xb
	bl	calloc
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	bl	calloc
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	bl	calloc
	adrp	x5, .LCPI0_0
	adrp	x6, .LCPI0_1
	adrp	x7, .LCPI0_3
	ldr	q0, [x5, :lo12:.LCPI0_0]
	mov	x5, x1
	ldr	q1, [x6, :lo12:.LCPI0_1]
	adrp	x6, .LCPI0_2
	mov	x3, x0
	mov	w10, wzr
	str	q0, [x5, #4]!
	ldr	d0, [x6, :lo12:.LCPI0_2]
	mov	w4, wzr
	mov	x2, xzr
	sub	x6, x0, #4
	stur	d0, [x1, #36]
	ldr	q0, [x7, :lo12:.LCPI0_3]
	mov	w8, #10                         // =0xa
	adrp	x7, .L.str.1
	add	x7, x7, :lo12:.L.str.1
	mov	w9, #10                         // =0xa
	stur	q1, [x1, #20]
.LBB0_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_9 Depth 2
                                        //     Child Loop BB0_16 Depth 2
                                        //       Child Loop BB0_18 Depth 3
	cmp	w4, #29
	b.gt	.LBB0_3
// %bb.2:                               //   in Loop: Header=BB0_1 Depth=1
	ldr	w12, [x1]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #4]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #8]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #12]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #16]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #20]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #24]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #28]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #32]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	ldr	w12, [x1, #36]
	mov	x0, x7
	add	w1, w12, #1
	bl	printf
	add	w1, w9, #1
	mov	x0, x7
	bl	printf
	mov	w0, #10                         // =0xa
	bl	putchar
	add	w4, w4, #1
.LBB0_3:                                //   in Loop: Header=BB0_1 Depth=1
	tbz	w10, #0, .LBB0_7
.LBB0_4:                                //   in Loop: Header=BB0_1 Depth=1
	ldr	w10, [x1]
	cbz	w10, .LBB0_20
// %bb.5:                               //   in Loop: Header=BB0_1 Depth=1
	cmp	w8, #10
	b.ne	.LBB0_13
// %bb.6:                               //   in Loop: Header=BB0_1 Depth=1
	mov	w9, #10                         // =0xa
	b	.LBB0_20
.LBB0_7:                                //   in Loop: Header=BB0_1 Depth=1
	mov	w13, w11
	mov	x10, xzr
	sub	x12, x13, #2
	and	x13, x13, #0xe
	dup	v1.2d, x12
	add	x12, x6, w11, uxtw #2
	b	.LBB0_9
.LBB0_8:                                //   in Loop: Header=BB0_9 Depth=2
	add	x10, x10, #2
	sub	w11, w11, #2
	sub	x12, x12, #8
	cmp	x13, x10
	b.eq	.LBB0_4
.LBB0_9:                                //   Parent Loop BB0_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dup	v2.2d, x10
	orr	v2.16b, v2.16b, v0.16b
	cmhs	v2.2d, v1.2d, v2.2d
	xtn	v2.2s, v2.2d
	fmov	w14, s2
	tbnz	w14, #0, .LBB0_11
// %bb.10:                              //   in Loop: Header=BB0_9 Depth=2
	mov	w14, v2.s[1]
	tbz	w14, #0, .LBB0_8
	b	.LBB0_12
.LBB0_11:                               //   in Loop: Header=BB0_9 Depth=2
	str	w11, [x12]
	mov	w14, v2.s[1]
	tbz	w14, #0, .LBB0_8
.LBB0_12:                               //   in Loop: Header=BB0_9 Depth=2
	sub	w14, w11, #1
	stur	w14, [x12, #-4]
	b	.LBB0_8
.LBB0_13:                               //   in Loop: Header=BB0_1 Depth=1
	ldp	q1, q2, [x5]
	mov	x9, xzr
	ldr	x11, [x5, #32]
	stur	x11, [x0, #36]
	mov	w11, w10
	stur	q1, [x0, #4]
	stur	q2, [x0, #20]
	b	.LBB0_16
.LBB0_14:                               //   in Loop: Header=BB0_16 Depth=2
	sxtw	x12, w11
.LBB0_15:                               //   in Loop: Header=BB0_16 Depth=2
	ldr	w13, [x0, x12, lsl #2]
	add	x9, x9, #1
	str	w11, [x0, x12, lsl #2]
	mov	w11, w13
	cbz	w13, .LBB0_19
.LBB0_16:                               //   Parent Loop BB0_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_18 Depth 3
	cmp	w11, #2
	b.le	.LBB0_14
// %bb.17:                              //   in Loop: Header=BB0_16 Depth=2
	mov	w12, w11
	mov	w14, #1                         // =0x1
	sub	x13, x12, #1
.LBB0_18:                               //   Parent Loop BB0_1 Depth=1
                                        //     Parent Loop BB0_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	w16, [x0, x13, lsl #2]
	ldr	w15, [x0, x14, lsl #2]
	str	w16, [x0, x14, lsl #2]
	add	x14, x14, #1
	str	w15, [x0, x13, lsl #2]
	sub	x13, x13, #1
	cmp	x14, x13
	b.lt	.LBB0_18
	b	.LBB0_15
.LBB0_19:                               //   in Loop: Header=BB0_1 Depth=1
	cmp	x2, x9
	csel	x2, x2, x9, gt
	mov	w9, w8
.LBB0_20:                               //   in Loop: Header=BB0_1 Depth=1
	ldr	w13, [x1, #4]
	ldr	w11, [x3, #4]
	stp	w13, w10, [x1]
	subs	w10, w11, #1
	mov	w11, #1                         // =0x1
	str	w10, [x3, #4]
	cset	w10, gt
	b.gt	.LBB0_1
// %bb.21:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w11, [x3, #8]
	ldur	x12, [x1, #4]
	str	w13, [x1, #8]
	subs	w11, w11, #1
	str	x12, [x1]
	str	w11, [x3, #8]
	mov	w11, #2                         // =0x2
	b.gt	.LBB0_1
// %bb.22:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	x11, [x5]
	ldr	w14, [x3, #12]
	ldr	w13, [x5, #8]
	str	x11, [x1]
	subs	w11, w14, #1
	str	w11, [x3, #12]
	mov	w11, #3                         // =0x3
	stp	w13, w12, [x1, #8]
	b.gt	.LBB0_1
// %bb.23:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w11, [x1]
	ldr	w12, [x3, #16]
	ldr	q1, [x5]
	str	w11, [x1, #16]
	subs	w11, w12, #1
	str	w11, [x3, #16]
	mov	w11, #4                         // =0x4
	str	q1, [x1]
	b.gt	.LBB0_1
// %bb.24:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w12, [x5, #16]
	ldr	w11, [x1]
	ldr	w13, [x3, #20]
	ldr	q1, [x5]
	stp	w12, w11, [x1, #16]
	subs	w12, w13, #1
	mov	w11, #5                         // =0x5
	str	q1, [x1]
	str	w12, [x3, #20]
	b.gt	.LBB0_1
// %bb.25:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	x12, [x5, #16]
	ldr	w11, [x1]
	ldr	w13, [x3, #24]
	ldr	q1, [x5]
	str	x12, [x1, #16]
	subs	w12, w13, #1
	str	w11, [x1, #24]
	mov	w11, #6                         // =0x6
	str	q1, [x1]
	str	w12, [x3, #24]
	b.gt	.LBB0_1
// %bb.26:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w12, [x5, #24]
	ldr	w11, [x1]
	ldr	x13, [x5, #16]
	ldr	q1, [x5]
	stp	w12, w11, [x1, #24]
	ldr	w12, [x3, #28]
	mov	w11, #7                         // =0x7
	str	x13, [x1, #16]
	subs	w12, w12, #1
	str	q1, [x1]
	str	w12, [x3, #28]
	b.gt	.LBB0_1
// %bb.27:                              //   in Loop: Header=BB0_1 Depth=1
	ldp	q1, q2, [x5]
	ldr	w11, [x1]
	ldr	w12, [x3, #32]
	str	w11, [x1, #32]
	mov	w11, #8                         // =0x8
	subs	w12, w12, #1
	stp	q1, q2, [x1]
	str	w12, [x3, #32]
	b.gt	.LBB0_1
// %bb.28:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w12, [x5, #32]
	ldr	w11, [x1]
	ldp	q2, q1, [x5]
	stp	w12, w11, [x1, #32]
	ldr	w12, [x3, #36]
	mov	w11, #9                         // =0x9
	subs	w12, w12, #1
	stp	q2, q1, [x1]
	str	w12, [x3, #36]
	b.gt	.LBB0_1
// %bb.29:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [x5, #32]
	ldp	q2, q1, [x5]
	ldr	w9, [x3, #40]
	mov	w11, #10                        // =0xa
	str	x8, [x1, #32]
	ldr	w8, [x1]
	subs	w12, w9, #1
	stp	q2, q1, [x1]
	mov	w9, w8
	str	w8, [x1, #40]
	str	w12, [x3, #40]
	b.gt	.LBB0_1
// %bb.30:
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, #11                         // =0xb
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Pfannkuchen(%d) = %ld\n"
	.size	.L.str, 23

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"%d"
	.size	.L.str.1, 3

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
