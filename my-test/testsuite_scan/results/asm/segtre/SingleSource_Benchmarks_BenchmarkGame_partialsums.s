	.file	"partialsums.c"
	.text
	.globl	make_vec                        // -- Begin function make_vec
	.p2align	2
	.type	make_vec,@function
make_vec:                               // @make_vec
	.cfi_startproc
// %bb.0:
	fmov	d0, d1
	mov	v1.d[1], v0.d[0]
	mov	v0.16b, v1.16b
	ret
.Lfunc_end0:
	.size	make_vec, .Lfunc_end0-make_vec
	.cfi_endproc
                                        // -- End function
	.globl	sum_vec                         // -- Begin function sum_vec
	.p2align	2
	.type	sum_vec,@function
sum_vec:                                // @sum_vec
	.cfi_startproc
// %bb.0:
	faddp	d0, v0.2d
	ret
.Lfunc_end1:
	.size	sum_vec, .Lfunc_end1-sum_vec
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function main
.LCPI2_0:
	.xword	0x3ff0000000000000              // double 1
	.xword	0x4000000000000000              // double 2
.LCPI2_1:
	.xword	0x3ff0000000000000              // double 1
	.xword	0xbff0000000000000              // double -1
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
	mov	x1, #6148914691236517205        // =0x5555555555555555
	movi	d2, #0000000000000000
	movi	d3, #0000000000000000
	movi	d1, #0000000000000000
	movi	d0, #0000000000000000
	movk	x1, #16357, lsl #48
	fmov	d4, #1.00000000
	fmov	d5, #1.00000000
	mov	w0, wzr
	fmov	d6, x1
	mov	w1, #9632                       // =0x25a0
	movk	w1, #38, lsl #16
.LBB2_1:                                // =>This Inner Loop Header: Depth=1
	scvtf	d1, w0
	fmov	d0, d6
	bl	pow
	fsqrt	d8, d5
	fmov	d7, d0
	fcmp	d8, d8
	b.vs	.LBB2_3
.LBB2_2:                                // %.split
                                        //   in Loop: Header=BB2_1 Depth=1
	fdiv	d8, d4, d8
	fmul	d9, d5, d5
	fmov	d0, d5
	fadd	d2, d2, d7
	fmul	d7, d9, d5
	fadd	d3, d3, d8
	bl	sin
	fmov	d8, d0
	fmov	d0, d5
	bl	cos
	fmul	d10, d7, d8
	fmul	d11, d7, d0
	fadd	d5, d5, d4
	add	w0, w0, #1
	cmp	w0, w1
	fmul	d7, d8, d10
	fmul	d9, d0, d11
	fdiv	d8, d4, d7
	fdiv	d7, d4, d9
	fadd	d1, d1, d8
	fadd	d0, d0, d7
	b.ne	.LBB2_1
	b	.LBB2_4
.LBB2_3:                                // %call.sqrt
                                        //   in Loop: Header=BB2_1 Depth=1
	fmov	d0, d5
	bl	sqrt
	fmov	d8, d0
	b	.LBB2_2
.LBB2_4:                                // %.preheader
	adrp	x0, .LCPI2_1
	movi	v6.2d, #0000000000000000
	fmov	v9.2d, #1.00000000
	ldr	q13, [x0, :lo12:.LCPI2_1]
	mov	x0, #20684562497536             // =0x12d000000000
	fmov	v10.2d, #2.00000000
	fmov	v11.2d, #-1.00000000
	movi	v7.2d, #0000000000000000
	movi	v8.2d, #0000000000000000
	movi	v5.2d, #0000000000000000
	movi	v4.2d, #0000000000000000
	movk	x0, #16707, lsl #48
	adrp	x1, .LCPI2_0
	fmov	d14, x0
	ldr	q12, [x1, :lo12:.LCPI2_0]
.LBB2_5:                                // =>This Inner Loop Header: Depth=1
	fadd	v15.2d, v12.2d, v9.2d
	mov	v19.16b, v11.16b
	fdiv	v16.2d, v9.2d, v12.2d
	fmul	v17.2d, v12.2d, v12.2d
	fmla	v19.2d, v10.2d, v12.2d
	fmul	v15.2d, v12.2d, v15.2d
	fdiv	v15.2d, v9.2d, v15.2d
	fadd	v6.2d, v6.2d, v16.2d
	fdiv	v17.2d, v9.2d, v17.2d
	fadd	v8.2d, v8.2d, v15.2d
	fdiv	v18.2d, v13.2d, v12.2d
	fadd	v12.2d, v12.2d, v10.2d
	fadd	v7.2d, v7.2d, v17.2d
	fcmp	d12, d14
	fdiv	v19.2d, v13.2d, v19.2d
	fadd	v5.2d, v5.2d, v18.2d
	fadd	v4.2d, v4.2d, v19.2d
	b.ls	.LBB2_5
// %bb.6:
	fmov	d0, d2
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	adrp	x1, .L.str.1
	add	x1, x1, :lo12:.L.str.1
	bl	printf
	fmov	d0, d3
	adrp	x1, .L.str.2
	add	x1, x1, :lo12:.L.str.2
	bl	printf
	faddp	d0, v8.2d
	adrp	x1, .L.str.3
	add	x1, x1, :lo12:.L.str.3
	bl	printf
	fmov	d0, d1
	adrp	x1, .L.str.4
	add	x1, x1, :lo12:.L.str.4
	bl	printf
	adrp	x1, .L.str.5
	add	x1, x1, :lo12:.L.str.5
	bl	printf
	faddp	d0, v6.2d
	adrp	x1, .L.str.6
	add	x1, x1, :lo12:.L.str.6
	bl	printf
	faddp	d0, v7.2d
	adrp	x1, .L.str.7
	add	x1, x1, :lo12:.L.str.7
	bl	printf
	faddp	d0, v5.2d
	adrp	x1, .L.str.8
	add	x1, x1, :lo12:.L.str.8
	bl	printf
	faddp	d0, v4.2d
	adrp	x1, .L.str.9
	add	x1, x1, :lo12:.L.str.9
	bl	printf
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
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%.9f\t%s\n"
	.size	.L.str, 9

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"(2/3)^k"
	.size	.L.str.1, 8

	.type	.L.str.2,@object                // @.str.2
.L.str.2:
	.asciz	"k^-0.5"
	.size	.L.str.2, 7

	.type	.L.str.3,@object                // @.str.3
.L.str.3:
	.asciz	"1/k(k+1)"
	.size	.L.str.3, 9

	.type	.L.str.4,@object                // @.str.4
.L.str.4:
	.asciz	"Flint Hills"
	.size	.L.str.4, 12

	.type	.L.str.5,@object                // @.str.5
.L.str.5:
	.asciz	"Cookson Hills"
	.size	.L.str.5, 14

	.type	.L.str.6,@object                // @.str.6
.L.str.6:
	.asciz	"Harmonic"
	.size	.L.str.6, 9

	.type	.L.str.7,@object                // @.str.7
.L.str.7:
	.asciz	"Riemann Zeta"
	.size	.L.str.7, 13

	.type	.L.str.8,@object                // @.str.8
.L.str.8:
	.asciz	"Alternating Harmonic"
	.size	.L.str.8, 21

	.type	.L.str.9,@object                // @.str.9
.L.str.9:
	.asciz	"Gregory"
	.size	.L.str.9, 8

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
