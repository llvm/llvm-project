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
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #48]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #64]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #80]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #96]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            // 16-byte Folded Spill
	add	x29, sp, #32
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
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	mov	w21, #11                        // =0xb
	bl	calloc
	mov	x19, x0
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	bl	calloc
	mov	x20, x0
	mov	w0, #11                         // =0xb
	mov	w1, #4                          // =0x4
	bl	calloc
	adrp	x8, .LCPI0_0
	mov	x25, x20
	mov	x22, x0
	ldr	q0, [x8, :lo12:.LCPI0_0]
	adrp	x8, .LCPI0_1
	mov	w26, wzr
	ldr	q1, [x8, :lo12:.LCPI0_1]
	adrp	x8, .LCPI0_2
	mov	w24, wzr
	str	q0, [x25, #4]!
	ldr	d0, [x8, :lo12:.LCPI0_2]
	adrp	x8, .LCPI0_3
	sub	x9, x0, #4
	mov	w27, #10                        // =0xa
	ldr	q2, [x8, :lo12:.LCPI0_3]
	adrp	x23, .L.str.1
	add	x23, x23, :lo12:.L.str.1
	mov	w28, #10                        // =0xa
	stur	xzr, [x29, #-8]                 // 8-byte Folded Spill
	stur	q1, [x20, #20]
	stur	d0, [x20, #36]
	str	x9, [sp, #16]                   // 8-byte Folded Spill
	str	q2, [sp]                        // 16-byte Folded Spill
.LBB0_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_9 Depth 2
                                        //     Child Loop BB0_16 Depth 2
                                        //       Child Loop BB0_18 Depth 3
	cmp	w24, #29
	b.gt	.LBB0_3
// %bb.2:                               //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x20]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #4]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #8]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #12]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #16]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #20]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #24]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #28]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #32]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	ldr	w8, [x20, #36]
	mov	x0, x23
	add	w1, w8, #1
	bl	printf
	add	w1, w28, #1
	mov	x0, x23
	bl	printf
	mov	w0, #10                         // =0xa
	bl	putchar
	ldr	q2, [sp]                        // 16-byte Folded Reload
	add	w24, w24, #1
.LBB0_3:                                //   in Loop: Header=BB0_1 Depth=1
	tbz	w26, #0, .LBB0_7
.LBB0_4:                                //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x20]
	cbz	w8, .LBB0_20
// %bb.5:                               //   in Loop: Header=BB0_1 Depth=1
	cmp	w27, #10
	b.ne	.LBB0_13
// %bb.6:                               //   in Loop: Header=BB0_1 Depth=1
	mov	w28, #10                        // =0xa
	b	.LBB0_20
.LBB0_7:                                //   in Loop: Header=BB0_1 Depth=1
	mov	w10, w21
	mov	x8, xzr
	sub	x9, x10, #2
	and	x10, x10, #0xe
	dup	v0.2d, x9
	ldr	x9, [sp, #16]                   // 8-byte Folded Reload
	add	x9, x9, w21, uxtw #2
	b	.LBB0_9
.LBB0_8:                                //   in Loop: Header=BB0_9 Depth=2
	add	x8, x8, #2
	sub	w21, w21, #2
	sub	x9, x9, #8
	cmp	x10, x8
	b.eq	.LBB0_4
.LBB0_9:                                //   Parent Loop BB0_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	dup	v1.2d, x8
	orr	v1.16b, v1.16b, v2.16b
	cmhs	v1.2d, v0.2d, v1.2d
	xtn	v1.2s, v1.2d
	fmov	w11, s1
	tbnz	w11, #0, .LBB0_11
// %bb.10:                              //   in Loop: Header=BB0_9 Depth=2
	mov	w11, v1.s[1]
	tbz	w11, #0, .LBB0_8
	b	.LBB0_12
.LBB0_11:                               //   in Loop: Header=BB0_9 Depth=2
	str	w21, [x9]
	mov	w11, v1.s[1]
	tbz	w11, #0, .LBB0_8
.LBB0_12:                               //   in Loop: Header=BB0_9 Depth=2
	sub	w11, w21, #1
	stur	w11, [x9, #-4]
	b	.LBB0_8
.LBB0_13:                               //   in Loop: Header=BB0_1 Depth=1
	ldp	q0, q1, [x25]
	mov	x9, xzr
	ldr	x10, [x25, #32]
	stur	x10, [x19, #36]
	mov	w10, w8
	stur	q0, [x19, #4]
	stur	q1, [x19, #20]
	b	.LBB0_16
.LBB0_14:                               //   in Loop: Header=BB0_16 Depth=2
	sxtw	x11, w10
.LBB0_15:                               //   in Loop: Header=BB0_16 Depth=2
	ldr	w12, [x19, x11, lsl #2]
	add	x9, x9, #1
	str	w10, [x19, x11, lsl #2]
	mov	w10, w12
	cbz	w12, .LBB0_19
.LBB0_16:                               //   Parent Loop BB0_1 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_18 Depth 3
	cmp	w10, #2
	b.le	.LBB0_14
// %bb.17:                              //   in Loop: Header=BB0_16 Depth=2
	mov	w11, w10
	mov	w13, #1                         // =0x1
	sub	x12, x11, #1
.LBB0_18:                               //   Parent Loop BB0_1 Depth=1
                                        //     Parent Loop BB0_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	w14, [x19, x12, lsl #2]
	ldr	w15, [x19, x13, lsl #2]
	str	w14, [x19, x13, lsl #2]
	add	x13, x13, #1
	str	w15, [x19, x12, lsl #2]
	sub	x12, x12, #1
	cmp	x13, x12
	b.lt	.LBB0_18
	b	.LBB0_15
.LBB0_19:                               //   in Loop: Header=BB0_1 Depth=1
	ldur	x10, [x29, #-8]                 // 8-byte Folded Reload
	mov	w28, w27
	cmp	x10, x9
	csel	x10, x10, x9, gt
	stur	x10, [x29, #-8]                 // 8-byte Folded Spill
.LBB0_20:                               //   in Loop: Header=BB0_1 Depth=1
	ldr	w9, [x20, #4]
	ldr	w10, [x22, #4]
	mov	w21, #1                         // =0x1
	stp	w9, w8, [x20]
	subs	w8, w10, #1
	cset	w26, gt
	str	w8, [x22, #4]
	b.gt	.LBB0_1
// %bb.21:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w10, [x22, #8]
	ldur	x8, [x20, #4]
	mov	w21, #2                         // =0x2
	str	w9, [x20, #8]
	subs	w9, w10, #1
	str	x8, [x20]
	str	w9, [x22, #8]
	b.gt	.LBB0_1
// %bb.22:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w10, [x25, #8]
	ldr	w11, [x22, #12]
	mov	w21, #3                         // =0x3
	ldr	x9, [x25]
	stp	w10, w8, [x20, #8]
	subs	w8, w11, #1
	str	x9, [x20]
	str	w8, [x22, #12]
	b.gt	.LBB0_1
// %bb.23:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x20]
	ldr	w9, [x22, #16]
	mov	w21, #4                         // =0x4
	ldr	q0, [x25]
	str	w8, [x20, #16]
	subs	w8, w9, #1
	str	q0, [x20]
	str	w8, [x22, #16]
	b.gt	.LBB0_1
// %bb.24:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x25, #16]
	ldr	w9, [x20]
	mov	w21, #5                         // =0x5
	ldr	w10, [x22, #20]
	ldr	q0, [x25]
	stp	w8, w9, [x20, #16]
	subs	w8, w10, #1
	str	q0, [x20]
	str	w8, [x22, #20]
	b.gt	.LBB0_1
// %bb.25:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [x25, #16]
	ldr	w10, [x22, #24]
	mov	w21, #6                         // =0x6
	ldr	q0, [x25]
	ldr	w9, [x20]
	str	x8, [x20, #16]
	subs	w8, w10, #1
	str	q0, [x20]
	str	w9, [x20, #24]
	str	w8, [x22, #24]
	b.gt	.LBB0_1
// %bb.26:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x25, #24]
	ldr	w10, [x20]
	mov	w21, #7                         // =0x7
	ldr	x9, [x25, #16]
	ldr	q0, [x25]
	stp	w8, w10, [x20, #24]
	ldr	w8, [x22, #28]
	str	x9, [x20, #16]
	subs	w8, w8, #1
	str	q0, [x20]
	str	w8, [x22, #28]
	b.gt	.LBB0_1
// %bb.27:                              //   in Loop: Header=BB0_1 Depth=1
	ldp	q1, q0, [x25]
	mov	w21, #8                         // =0x8
	ldr	w9, [x22, #32]
	ldr	w8, [x20]
	subs	w9, w9, #1
	stp	q1, q0, [x20]
	str	w8, [x20, #32]
	str	w9, [x22, #32]
	b.gt	.LBB0_1
// %bb.28:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	w8, [x25, #32]
	ldr	w9, [x20]
	mov	w21, #9                         // =0x9
	ldp	q1, q0, [x25]
	stp	w8, w9, [x20, #32]
	ldr	w8, [x22, #36]
	subs	w8, w8, #1
	stp	q1, q0, [x20]
	str	w8, [x22, #36]
	b.gt	.LBB0_1
// %bb.29:                              //   in Loop: Header=BB0_1 Depth=1
	ldr	x8, [x25, #32]
	ldp	q1, q0, [x25]
	ldr	w27, [x20]
	mov	w21, #10                        // =0xa
	str	x8, [x20, #32]
	ldr	w8, [x22, #40]
	mov	w28, w27
	stp	q1, q0, [x20]
	subs	w8, w8, #1
	str	w27, [x20, #40]
	str	w8, [x22, #40]
	b.gt	.LBB0_1
// %bb.30:
	ldur	x2, [x29, #-8]                  // 8-byte Folded Reload
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, #11                         // =0xb
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #128
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
