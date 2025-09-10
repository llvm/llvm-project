	.file	"partialsums.c"
	.text
	.globl	make_vec                        // -- Begin function make_vec
	.p2align	2
	.type	make_vec,@function
make_vec:                               // @make_vec
	.cfi_startproc
// %bb.0:
                                        // kill: def $d0 killed $d0 def $q0
                                        // kill: def $d1 killed $d1 def $q1
	mov	v0.d[1], v1.d[0]
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
	sub	sp, sp, #176
	.cfi_def_cfa_offset 176
	stp	d15, d14, [sp, #80]             // 16-byte Folded Spill
	stp	d13, d12, [sp, #96]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #112]            // 16-byte Folded Spill
	stp	d9, d8, [sp, #128]              // 16-byte Folded Spill
	stp	x29, x30, [sp, #144]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #160]            // 16-byte Folded Spill
	add	x29, sp, #144
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
	movi	d10, #0000000000000000
	movi	d11, #0000000000000000
	mov	x8, #6148914691236517205        // =0x5555555555555555
	movi	d9, #0000000000000000
	movi	d8, #0000000000000000
	mov	w20, #9632                      // =0x25a0
	fmov	d15, #1.00000000
	fmov	d12, #1.00000000
	mov	w19, wzr
	movk	x8, #16357, lsl #48
	movk	w20, #38, lsl #16
	str	x8, [sp, #64]                   // 8-byte Folded Spill
.LBB2_1:                                // =>This Inner Loop Header: Depth=1
	scvtf	d1, w19
	ldr	d0, [sp, #64]                   // 8-byte Folded Reload
	bl	pow
	fsqrt	d1, d12
	fcmp	d1, d1
	b.vs	.LBB2_3
.LBB2_2:                                // %.split
                                        //   in Loop: Header=BB2_1 Depth=1
	fdiv	d1, d15, d1
	fmul	d2, d12, d12
	fadd	d10, d10, d0
	fmov	d0, d12
	fmul	d13, d2, d12
	fadd	d11, d11, d1
	bl	sin
	fmov	d14, d0
	fmov	d0, d12
	bl	cos
	fmul	d1, d13, d14
	fmul	d2, d13, d0
	fadd	d12, d12, d15
	add	w19, w19, #1
	cmp	w19, w20
	fmul	d1, d14, d1
	fmul	d0, d0, d2
	fdiv	d1, d15, d1
	fdiv	d0, d15, d0
	fadd	d9, d9, d1
	fadd	d8, d8, d0
	b.ne	.LBB2_1
	b	.LBB2_4
.LBB2_3:                                // %call.sqrt
                                        //   in Loop: Header=BB2_1 Depth=1
	fmov	d14, d0
	fmov	d0, d12
	bl	sqrt
	fmov	d1, d0
	fmov	d0, d14
	b	.LBB2_2
.LBB2_4:                                // %.preheader
	adrp	x8, .LCPI2_0
	movi	v21.2d, #0000000000000000
	fmov	v0.2d, #1.00000000
	ldr	q3, [x8, :lo12:.LCPI2_0]
	mov	x8, #20684562497536             // =0x12d000000000
	fmov	v1.2d, #2.00000000
	fmov	v2.2d, #-1.00000000
	movi	v22.2d, #0000000000000000
	movi	v23.2d, #0000000000000000
	movi	v20.2d, #0000000000000000
	movi	v19.2d, #0000000000000000
	movk	x8, #16707, lsl #48
	adrp	x9, .LCPI2_1
	fmov	d5, x8
	ldr	q4, [x9, :lo12:.LCPI2_1]
.LBB2_5:                                // =>This Inner Loop Header: Depth=1
	fadd	v6.2d, v3.2d, v0.2d
	mov	v18.16b, v2.16b
	fdiv	v7.2d, v0.2d, v3.2d
	fmul	v16.2d, v3.2d, v3.2d
	fmla	v18.2d, v1.2d, v3.2d
	fmul	v6.2d, v3.2d, v6.2d
	fdiv	v6.2d, v0.2d, v6.2d
	fadd	v21.2d, v21.2d, v7.2d
	fdiv	v16.2d, v0.2d, v16.2d
	fadd	v23.2d, v23.2d, v6.2d
	fdiv	v17.2d, v4.2d, v3.2d
	fadd	v3.2d, v3.2d, v1.2d
	fadd	v22.2d, v22.2d, v16.2d
	fcmp	d3, d5
	fdiv	v18.2d, v4.2d, v18.2d
	fadd	v20.2d, v20.2d, v17.2d
	fadd	v19.2d, v19.2d, v18.2d
	b.ls	.LBB2_5
// %bb.6:
	fmov	d0, d10
	adrp	x19, .L.str
	add	x19, x19, :lo12:.L.str
	adrp	x1, .L.str.1
	add	x1, x1, :lo12:.L.str.1
	mov	x0, x19
	stp	q21, q22, [sp, #16]             // 32-byte Folded Spill
	stp	q20, q19, [sp, #48]             // 32-byte Folded Spill
	str	q23, [sp]                       // 16-byte Folded Spill
	bl	printf
	fmov	d0, d11
	adrp	x1, .L.str.2
	add	x1, x1, :lo12:.L.str.2
	mov	x0, x19
	bl	printf
	ldr	q0, [sp]                        // 16-byte Folded Reload
	adrp	x1, .L.str.3
	add	x1, x1, :lo12:.L.str.3
	mov	x0, x19
	faddp	d0, v0.2d
	bl	printf
	fmov	d0, d9
	adrp	x1, .L.str.4
	add	x1, x1, :lo12:.L.str.4
	mov	x0, x19
	bl	printf
	fmov	d0, d8
	adrp	x1, .L.str.5
	add	x1, x1, :lo12:.L.str.5
	mov	x0, x19
	bl	printf
	ldr	q0, [sp, #16]                   // 16-byte Folded Reload
	adrp	x1, .L.str.6
	add	x1, x1, :lo12:.L.str.6
	mov	x0, x19
	faddp	d0, v0.2d
	bl	printf
	ldr	q0, [sp, #32]                   // 16-byte Folded Reload
	adrp	x1, .L.str.7
	add	x1, x1, :lo12:.L.str.7
	mov	x0, x19
	faddp	d0, v0.2d
	bl	printf
	ldr	q0, [sp, #48]                   // 16-byte Folded Reload
	adrp	x1, .L.str.8
	add	x1, x1, :lo12:.L.str.8
	mov	x0, x19
	faddp	d0, v0.2d
	bl	printf
	ldr	q0, [sp, #64]                   // 16-byte Folded Reload
	adrp	x1, .L.str.9
	add	x1, x1, :lo12:.L.str.9
	mov	x0, x19
	faddp	d0, v0.2d
	bl	printf
	mov	w0, wzr
	.cfi_def_cfa wsp, 176
	ldp	x20, x19, [sp, #160]            // 16-byte Folded Reload
	ldp	x29, x30, [sp, #144]            // 16-byte Folded Reload
	ldp	d9, d8, [sp, #128]              // 16-byte Folded Reload
	ldp	d11, d10, [sp, #112]            // 16-byte Folded Reload
	ldp	d13, d12, [sp, #96]             // 16-byte Folded Reload
	ldp	d15, d14, [sp, #80]             // 16-byte Folded Reload
	add	sp, sp, #176
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
