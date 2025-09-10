	.file	"recursive.c"
	.text
	.globl	ack                             // -- Begin function ack
	.p2align	2
	.type	ack,@function
ack:                                    // @ack
	.cfi_startproc
// %bb.0:
	cbz	w0, .LBB0_6
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	b	.LBB0_3
.LBB0_2:                                //   in Loop: Header=BB0_3 Depth=1
	sub	w1, w1, #1
	bl	ack
	mov	w1, w0
	sub	w0, w0, #1
	cbz	w0, .LBB0_5
.LBB0_3:                                // =>This Inner Loop Header: Depth=1
	cbnz	w1, .LBB0_2
// %bb.4:                               //   in Loop: Header=BB0_3 Depth=1
	mov	w1, #1                          // =0x1
	sub	w0, w0, #1
	cbnz	w0, .LBB0_3
.LBB0_5:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB0_6:
	add	w0, w1, #1
	ret
.Lfunc_end0:
	.size	ack, .Lfunc_end0-ack
	.cfi_endproc
                                        // -- End function
	.globl	fib                             // -- Begin function fib
	.p2align	2
	.type	fib,@function
fib:                                    // @fib
	.cfi_startproc
// %bb.0:
	cmp	w0, #2
	b.ge	.LBB1_2
// %bb.1:
	mov	w0, #1                          // =0x1
	ret
.LBB1_2:                                // %.preheader
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w1, w0
	mov	w0, wzr
	add	w1, w1, #1
.LBB1_3:                                // =>This Inner Loop Header: Depth=1
	sub	w0, w1, #3
	bl	fib
	sub	w1, w1, #1
	add	w0, w0, w0
	cmp	w1, #2
	b.hi	.LBB1_3
// %bb.4:
	add	w0, w0, #1
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end1:
	.size	fib, .Lfunc_end1-fib
	.cfi_endproc
                                        // -- End function
	.globl	fibFP                           // -- Begin function fibFP
	.p2align	2
	.type	fibFP,@function
fibFP:                                  // @fibFP
	.cfi_startproc
// %bb.0:
	fmov	d1, #2.00000000
	fcmp	d0, d1
	fmov	d1, #1.00000000
	b.mi	.LBB2_2
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	fmov	d1, #-2.00000000
	fadd	d0, d0, d1
	bl	fibFP
	fmov	d1, #-1.00000000
	fadd	d0, d0, d1
	bl	fibFP
	fadd	d1, d0, d0
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB2_2:
	fmov	d0, d1
	ret
.Lfunc_end2:
	.size	fibFP, .Lfunc_end2-fibFP
	.cfi_endproc
                                        // -- End function
	.globl	tak                             // -- Begin function tak
	.p2align	2
	.type	tak,@function
tak:                                    // @tak
	.cfi_startproc
// %bb.0:
	cmp	w1, w0
	b.ge	.LBB3_4
// %bb.1:                               // %.preheader
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w2, w0
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	sub	w0, w2, #1
	mov	w2, w0
	bl	tak
	sub	w0, w1, #1
	mov	w1, w0
	bl	tak
	sub	w0, w0, #1
	mov	w1, w2
	bl	tak
	cmp	w4, w3
	mov	w1, w4
	mov	w2, w3
	b.lt	.LBB3_2
// %bb.3:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB3_4:
	ret
.Lfunc_end3:
	.size	tak, .Lfunc_end3-tak
	.cfi_endproc
                                        // -- End function
	.globl	takFP                           // -- Begin function takFP
	.p2align	2
	.type	takFP,@function
takFP:                                  // @takFP
	.cfi_startproc
// %bb.0:
	fmov	d2, d1
	fmov	d3, d0
	fcmp	d1, d0
	fmov	d0, d2
	b.pl	.LBB4_4
// %bb.1:                               // %.preheader
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	fmov	d1, #-1.00000000
.LBB4_2:                                // =>This Inner Loop Header: Depth=1
	fadd	d0, d3, d1
	fmov	d1, d2
	fmov	d2, d0
	bl	takFP
	fadd	d5, d2, d1
	fmov	d2, d3
	fmov	d0, d5
	fmov	d1, d5
	bl	takFP
	fadd	d0, d0, d1
	fmov	d1, d3
	bl	takFP
	fmov	d2, d5
	fmov	d3, d4
	fcmp	d5, d4
	b.mi	.LBB4_2
// %bb.3:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB4_4:
	ret
.Lfunc_end4:
	.size	takFP, .Lfunc_end4-takFP
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
	mov	w0, #3                          // =0x3
	mov	w1, #11                         // =0xb
	bl	ack
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, #11                         // =0xb
	mov	w2, w0
	bl	printf
	mov	x0, #4630544841867001856        // =0x4043000000000000
	fmov	d0, x0
	bl	fibFP
	fmov	d1, d0
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	bl	printf
	mov	w0, #30                         // =0x1e
	mov	w1, #20                         // =0x14
	mov	w2, #10                         // =0xa
	bl	tak
	adrp	x0, .L.str.2
	add	x0, x0, :lo12:.L.str.2
	mov	w1, #30                         // =0x1e
	mov	w2, #20                         // =0x14
	mov	w3, #10                         // =0xa
	mov	w4, w0
	bl	printf
	mov	w0, #3                          // =0x3
	bl	fib
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	mov	w1, w0
	bl	printf
	fmov	d0, #3.00000000
	fmov	d1, #2.00000000
	fmov	d2, #1.00000000
	bl	takFP
	adrp	x0, .L.str.4
	add	x0, x0, :lo12:.L.str.4
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end5:
	.size	main, .Lfunc_end5-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Ack(3,%d): %d\n"
	.size	.L.str, 15

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"Fib(%.1f): %.1f\n"
	.size	.L.str.1, 17

	.type	.L.str.2,@object                // @.str.2
.L.str.2:
	.asciz	"Tak(%d,%d,%d): %d\n"
	.size	.L.str.2, 19

	.type	.L.str.3,@object                // @.str.3
.L.str.3:
	.asciz	"Fib(3): %d\n"
	.size	.L.str.3, 12

	.type	.L.str.4,@object                // @.str.4
.L.str.4:
	.asciz	"Tak(3.0,2.0,1.0): %.1f\n"
	.size	.L.str.4, 24

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
