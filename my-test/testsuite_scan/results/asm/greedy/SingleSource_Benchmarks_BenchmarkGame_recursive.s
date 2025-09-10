	.file	"recursive.c"
	.text
	.globl	ack                             // -- Begin function ack
	.p2align	2
	.type	ack,@function
ack:                                    // @ack
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	str	x19, [sp, #16]                  // 8-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	mov	w19, w0
	mov	w0, w1
	cbnz	w19, .LBB0_2
	b	.LBB0_4
.LBB0_1:                                //   in Loop: Header=BB0_2 Depth=1
	sub	w1, w0, #1
	mov	w0, w19
	bl	ack
	sub	w19, w19, #1
	cbz	w19, .LBB0_4
.LBB0_2:                                // =>This Inner Loop Header: Depth=1
	cbnz	w0, .LBB0_1
// %bb.3:                               //   in Loop: Header=BB0_2 Depth=1
	mov	w0, #1                          // =0x1
	sub	w19, w19, #1
	cbnz	w19, .LBB0_2
.LBB0_4:
	add	w0, w0, #1
	.cfi_def_cfa wsp, 32
	ldr	x19, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w30
	.cfi_restore w29
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
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x20, x19, [sp, #16]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	mov	w19, wzr
	add	w20, w0, #1
.LBB1_3:                                // =>This Inner Loop Header: Depth=1
	sub	w0, w20, #3
	bl	fib
	sub	w20, w20, #1
	add	w19, w0, w19
	cmp	w20, #2
	b.hi	.LBB1_3
// %bb.4:
	add	w0, w19, #1
	.cfi_def_cfa wsp, 32
	ldp	x20, x19, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
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
	str	d8, [sp, #-32]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -32
	fmov	d1, #-2.00000000
	fmov	d8, d0
	fadd	d1, d0, d1
	fmov	d0, d1
	bl	fibFP
	fmov	d1, #-1.00000000
	fadd	d1, d8, d1
	fmov	d8, d0
	fmov	d0, d1
	bl	fibFP
	fadd	d1, d8, d0
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #32                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
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
	cmp	w1, w0
	mov	w19, w2
	b.ge	.LBB3_3
// %bb.1:                               // %.preheader
	mov	w20, w1
	mov	w21, w0
.LBB3_2:                                // =>This Inner Loop Header: Depth=1
	sub	w0, w21, #1
	mov	w1, w20
	mov	w2, w19
	bl	tak
	mov	w22, w0
	sub	w0, w20, #1
	mov	w1, w19
	mov	w2, w21
	bl	tak
	mov	w23, w0
	sub	w0, w19, #1
	mov	w1, w21
	mov	w2, w20
	bl	tak
	mov	w19, w0
	cmp	w23, w22
	mov	w20, w23
	mov	w21, w22
	b.lt	.LBB3_2
.LBB3_3:
	mov	w0, w19
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
	stp	d13, d12, [sp, #-64]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 64
	stp	d11, d10, [sp, #16]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #32]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #48]             // 16-byte Folded Spill
	add	x29, sp, #48
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -24
	.cfi_offset b9, -32
	.cfi_offset b10, -40
	.cfi_offset b11, -48
	.cfi_offset b12, -56
	.cfi_offset b13, -64
	fmov	d8, d2
	fcmp	d1, d0
	b.pl	.LBB4_3
// %bb.1:                               // %.preheader
	fmov	d9, d1
	fmov	d10, d0
	fmov	d13, #-1.00000000
.LBB4_2:                                // =>This Inner Loop Header: Depth=1
	fadd	d0, d10, d13
	fmov	d1, d9
	fmov	d2, d8
	bl	takFP
	fadd	d1, d9, d13
	fmov	d11, d0
	fmov	d2, d10
	fmov	d0, d1
	fmov	d1, d8
	bl	takFP
	fadd	d1, d8, d13
	fmov	d12, d0
	fmov	d2, d9
	fmov	d0, d1
	fmov	d1, d10
	bl	takFP
	fmov	d8, d0
	fmov	d9, d12
	fmov	d10, d11
	fcmp	d12, d11
	b.mi	.LBB4_2
.LBB4_3:
	fmov	d0, d8
	.cfi_def_cfa wsp, 64
	ldp	x29, x30, [sp, #48]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #32]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #16]             // 16-byte Folded Reload
	ldp	d13, d12, [sp], #64             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	.cfi_restore b13
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
	str	d8, [sp, #-32]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -32
	mov	w0, #3                          // =0x3
	mov	w1, #11                         // =0xb
	bl	ack
	mov	w2, w0
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, #11                         // =0xb
	bl	printf
	mov	x8, #4630544841867001856        // =0x4043000000000000
	fmov	d8, x8
	fmov	d0, d8
	bl	fibFP
	fmov	d1, d0
	fmov	d0, d8
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	bl	printf
	mov	w0, #30                         // =0x1e
	mov	w1, #20                         // =0x14
	mov	w2, #10                         // =0xa
	bl	tak
	mov	w4, w0
	adrp	x0, .L.str.2
	add	x0, x0, :lo12:.L.str.2
	mov	w1, #30                         // =0x1e
	mov	w2, #20                         // =0x14
	mov	w3, #10                         // =0xa
	bl	printf
	mov	w0, #3                          // =0x3
	bl	fib
	mov	w1, w0
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	bl	printf
	fmov	d0, #3.00000000
	fmov	d1, #2.00000000
	fmov	d2, #1.00000000
	bl	takFP
	adrp	x0, .L.str.4
	add	x0, x0, :lo12:.L.str.4
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #32                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
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
