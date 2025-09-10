	.file	"nsieve-bits.c"
	.text
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
	mov	w0, #8196                       // =0x2004
	movk	w0, #78, lsl #16
	bl	malloc
	mov	w1, #1                          // =0x1
	cbz	x0, .LBB0_26
// %bb.1:
	mov	w2, #32768                      // =0x8000
	mov	w1, #255                        // =0xff
	movk	w2, #312, lsl #16
	mov	w2, #8196                       // =0x2004
	movk	w2, #78, lsl #16
	bl	memset
	mov	w5, #1                          // =0x1
	mov	w3, wzr
	mov	w4, #2                          // =0x2
	movk	w5, #625, lsl #16
	mov	w6, #40960000                   // =0x2710000
	b	.LBB0_3
.LBB0_2:                                //   in Loop: Header=BB0_3 Depth=1
	add	w4, w4, #1
	cmp	w4, w5
	b.eq	.LBB0_9
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_7 Depth 2
	lsr	w7, w4, #5
	ldr	w7, [x0, w7, uxtw #2]
	lsr	w7, w7, w4
	tbz	w7, #0, .LBB0_2
// %bb.4:                               //   in Loop: Header=BB0_3 Depth=1
	cmp	w4, w2
	add	w3, w3, #1
	b.hi	.LBB0_2
// %bb.5:                               //   in Loop: Header=BB0_3 Depth=1
	lsl	w7, w4, #1
	b	.LBB0_7
.LBB0_6:                                //   in Loop: Header=BB0_7 Depth=2
	add	w7, w7, w4
	cmp	w7, w6
	b.hi	.LBB0_2
.LBB0_7:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w8, w7, #5
	lsl	w10, w1, w7
	ldr	w9, [x0, w8, uxtw #2]
	tst	w9, w10
	b.eq	.LBB0_6
// %bb.8:                               //   in Loop: Header=BB0_7 Depth=2
	eor	w9, w9, w10
	str	w9, [x0, w8, uxtw #2]
	b	.LBB0_6
.LBB0_9:
	mov	w1, #16384                      // =0x4000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	movk	w1, #156, lsl #16
	mov	w2, w3
	mov	w1, #40960000                   // =0x2710000
	bl	printf
	mov	w2, #4100                       // =0x1004
	mov	w1, #255                        // =0xff
	movk	w2, #39, lsl #16
	bl	memset
	mov	w3, wzr
	mov	w4, #2                          // =0x2
	mov	w5, #1                          // =0x1
	b	.LBB0_11
.LBB0_10:                               //   in Loop: Header=BB0_11 Depth=1
	cmp	w4, w2
	add	w4, w4, #1
	b.eq	.LBB0_17
.LBB0_11:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_15 Depth 2
	lsr	w6, w4, #5
	ldr	w6, [x0, w6, uxtw #2]
	lsr	w6, w6, w4
	tbz	w6, #0, .LBB0_10
// %bb.12:                              //   in Loop: Header=BB0_11 Depth=1
	cmp	w4, w1
	add	w3, w3, #1
	b.hi	.LBB0_10
// %bb.13:                              //   in Loop: Header=BB0_11 Depth=1
	lsl	w6, w4, #1
	b	.LBB0_15
.LBB0_14:                               //   in Loop: Header=BB0_15 Depth=2
	add	w6, w6, w4
	cmp	w6, w2
	b.hi	.LBB0_10
.LBB0_15:                               //   Parent Loop BB0_11 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w7, w6, #5
	lsl	w9, w5, w6
	ldr	w8, [x0, w7, uxtw #2]
	tst	w8, w9
	b.eq	.LBB0_14
// %bb.16:                              //   in Loop: Header=BB0_15 Depth=2
	eor	w8, w8, w9
	str	w8, [x0, w7, uxtw #2]
	b	.LBB0_14
.LBB0_17:
	mov	w1, #32768                      // =0x8000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	movk	w1, #312, lsl #16
	mov	w2, w3
	bl	printf
	mov	w2, #34820                      // =0x8804
	mov	w1, #255                        // =0xff
	movk	w2, #19, lsl #16
	bl	memset
	mov	w4, #16385                      // =0x4001
	mov	w2, wzr
	mov	w3, #2                          // =0x2
	movk	w4, #156, lsl #16
	sub	w5, w1, #1250, lsl #12          // =5120000
	mov	w6, #1                          // =0x1
	b	.LBB0_19
.LBB0_18:                               //   in Loop: Header=BB0_19 Depth=1
	add	w3, w3, #1
	cmp	w3, w4
	b.eq	.LBB0_25
.LBB0_19:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_23 Depth 2
	lsr	w7, w3, #5
	ldr	w7, [x0, w7, uxtw #2]
	lsr	w7, w7, w3
	tbz	w7, #0, .LBB0_18
// %bb.20:                              //   in Loop: Header=BB0_19 Depth=1
	cmp	w3, w5
	add	w2, w2, #1
	b.hi	.LBB0_18
// %bb.21:                              //   in Loop: Header=BB0_19 Depth=1
	lsl	w7, w3, #1
	b	.LBB0_23
.LBB0_22:                               //   in Loop: Header=BB0_23 Depth=2
	add	w7, w7, w3
	cmp	w7, w1
	b.hi	.LBB0_18
.LBB0_23:                               //   Parent Loop BB0_19 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsr	w8, w7, #5
	lsl	w10, w6, w7
	ldr	w9, [x0, w8, uxtw #2]
	tst	w9, w10
	b.eq	.LBB0_22
// %bb.24:                              //   in Loop: Header=BB0_23 Depth=2
	eor	w9, w9, w10
	str	w9, [x0, w8, uxtw #2]
	b	.LBB0_22
.LBB0_25:
	mov	w1, #16384                      // =0x4000
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	movk	w1, #156, lsl #16
	bl	printf
	bl	free
	mov	w1, wzr
.LBB0_26:
	mov	w0, w1
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
	.asciz	"Primes up to %8d %8d\n"
	.size	.L.str, 22

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
