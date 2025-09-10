	.file	"nsieve-bits.c"
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	str	x21, [sp, #16]                  // 8-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	w0, #8196                       // =0x2004
	movk	w0, #78, lsl #16
	bl	malloc
	mov	w20, #1                         // =0x1
	cbz	x0, .LBB0_26
// %bb.1:
	mov	w2, #8196                       // =0x2004
	mov	w21, #32768                     // =0x8000
	mov	w1, #255                        // =0xff
	movk	w2, #78, lsl #16
	mov	x19, x0
	movk	w21, #312, lsl #16
	bl	memset
	mov	w9, #1                          // =0x1
	mov	w2, wzr
	mov	w8, #2                          // =0x2
	movk	w9, #625, lsl #16
	mov	w10, #40960000                  // =0x2710000
	b	.LBB0_3
.LBB0_2:                                //   in Loop: Header=BB0_3 Depth=1
	add	w8, w8, #1
	cmp	w8, w9
	b.eq	.LBB0_9
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_7 Depth 2
	lsr	w11, w8, #5
	ldr	w11, [x19, w11, uxtw #2]
	lsr	w11, w11, w8
	tbz	w11, #0, .LBB0_2
// %bb.4:                               //   in Loop: Header=BB0_3 Depth=1
	cmp	w8, w21
	add	w2, w2, #1
	b.hi	.LBB0_2
// %bb.5:                               //   in Loop: Header=BB0_3 Depth=1
	lsl	w11, w8, #1
	b	.LBB0_7
.LBB0_6:                                //   in Loop: Header=BB0_7 Depth=2
	add	w11, w11, w8
	cmp	w11, w10
	b.hi	.LBB0_2
.LBB0_7:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w12, w11, #5
	lsl	w14, w20, w11
	ldr	w13, [x19, w12, uxtw #2]
	tst	w13, w14
	b.eq	.LBB0_6
// %bb.8:                               //   in Loop: Header=BB0_7 Depth=2
	eor	w13, w13, w14
	str	w13, [x19, w12, uxtw #2]
	b	.LBB0_6
.LBB0_9:
	mov	w20, #16384                     // =0x4000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, #40960000                   // =0x2710000
	movk	w20, #156, lsl #16
	bl	printf
	mov	w2, #4100                       // =0x1004
	mov	x0, x19
	mov	w1, #255                        // =0xff
	movk	w2, #39, lsl #16
	bl	memset
	mov	w2, wzr
	mov	w8, #2                          // =0x2
	mov	w9, #1                          // =0x1
	b	.LBB0_11
.LBB0_10:                               //   in Loop: Header=BB0_11 Depth=1
	cmp	w8, w21
	add	w8, w8, #1
	b.eq	.LBB0_17
.LBB0_11:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_15 Depth 2
	lsr	w10, w8, #5
	ldr	w10, [x19, w10, uxtw #2]
	lsr	w10, w10, w8
	tbz	w10, #0, .LBB0_10
// %bb.12:                              //   in Loop: Header=BB0_11 Depth=1
	cmp	w8, w20
	add	w2, w2, #1
	b.hi	.LBB0_10
// %bb.13:                              //   in Loop: Header=BB0_11 Depth=1
	lsl	w10, w8, #1
	b	.LBB0_15
.LBB0_14:                               //   in Loop: Header=BB0_15 Depth=2
	add	w10, w10, w8
	cmp	w10, w21
	b.hi	.LBB0_10
.LBB0_15:                               //   Parent Loop BB0_11 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w11, w10, #5
	lsl	w13, w9, w10
	ldr	w12, [x19, w11, uxtw #2]
	tst	w12, w13
	b.eq	.LBB0_14
// %bb.16:                              //   in Loop: Header=BB0_15 Depth=2
	eor	w12, w12, w13
	str	w12, [x19, w11, uxtw #2]
	b	.LBB0_14
.LBB0_17:
	mov	w1, #32768                      // =0x8000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	movk	w1, #312, lsl #16
	bl	printf
	mov	w2, #34820                      // =0x8804
	mov	x0, x19
	mov	w1, #255                        // =0xff
	movk	w2, #19, lsl #16
	bl	memset
	mov	w9, #16385                      // =0x4001
	mov	w2, wzr
	mov	w8, #2                          // =0x2
	movk	w9, #156, lsl #16
	sub	w10, w20, #1250, lsl #12        // =5120000
	mov	w11, #1                         // =0x1
	b	.LBB0_19
.LBB0_18:                               //   in Loop: Header=BB0_19 Depth=1
	add	w8, w8, #1
	cmp	w8, w9
	b.eq	.LBB0_25
.LBB0_19:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_23 Depth 2
	lsr	w12, w8, #5
	ldr	w12, [x19, w12, uxtw #2]
	lsr	w12, w12, w8
	tbz	w12, #0, .LBB0_18
// %bb.20:                              //   in Loop: Header=BB0_19 Depth=1
	cmp	w8, w10
	add	w2, w2, #1
	b.hi	.LBB0_18
// %bb.21:                              //   in Loop: Header=BB0_19 Depth=1
	lsl	w12, w8, #1
	b	.LBB0_23
.LBB0_22:                               //   in Loop: Header=BB0_23 Depth=2
	add	w12, w12, w8
	cmp	w12, w20
	b.hi	.LBB0_18
.LBB0_23:                               //   Parent Loop BB0_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w13, w12, #5
	lsl	w15, w11, w12
	ldr	w14, [x19, w13, uxtw #2]
	tst	w14, w15
	b.eq	.LBB0_22
// %bb.24:                              //   in Loop: Header=BB0_23 Depth=2
	eor	w14, w14, w15
	str	w14, [x19, w13, uxtw #2]
	b	.LBB0_22
.LBB0_25:
	mov	w1, #16384                      // =0x4000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	movk	w1, #156, lsl #16
	bl	printf
	mov	x0, x19
	bl	free
	mov	w20, wzr
.LBB0_26:
	mov	w0, w20
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x21, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
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
	.asciz	"Primes up to %8d %8d\n"
	.size	.L.str, 22

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
