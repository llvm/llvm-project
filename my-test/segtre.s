	.file	"massive_test.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function massive_vreg_test
.LCPI0_0:
	.word	99                              // 0x63
	.word	198                             // 0xc6
.LCPI0_1:
	.word	100                             // 0x64
	.word	199                             // 0xc7
.LCPI0_2:
	.word	297                             // 0x129
	.word	396                             // 0x18c
.LCPI0_3:
	.word	298                             // 0x12a
	.word	397                             // 0x18d
.LCPI0_4:
	.word	495                             // 0x1ef
	.word	594                             // 0x252
.LCPI0_5:
	.word	496                             // 0x1f0
	.word	595                             // 0x253
.LCPI0_6:
	.word	693                             // 0x2b5
	.word	792                             // 0x318
.LCPI0_7:
	.word	694                             // 0x2b6
	.word	793                             // 0x319
	.text
	.globl	massive_vreg_test
	.p2align	2
	.type	massive_vreg_test,@function
massive_vreg_test:                      // @massive_vreg_test
	.cfi_startproc
// %bb.0:                               // %entry
	str	d12, [sp, #-48]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	d11, d10, [sp, #16]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #32]               // 16-byte Folded Spill
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -48
	add	x2, x1, #396
	ldr	d1, [x1, #792]
	ldr	d0, [x1, #1584]
	ldr	d4, [x2]
	ldr	d2, [x2, #792]
	adrp	x4, .LCPI0_0
	adrp	x3, .LCPI0_1
	adrp	x5, .LCPI0_2
	ldr	d3, [x4, :lo12:.LCPI0_0]
	zip1	v6.2s, v4.2s, v1.2s
	zip2	v5.2s, v4.2s, v1.2s
	zip1	v9.2s, v2.2s, v0.2s
	ldr	d10, [x3, :lo12:.LCPI0_1]
	ldr	d11, [x5, :lo12:.LCPI0_2]
	ldr	d7, [x1, #2376]
	ldr	d8, [x2, #1584]
	zip2	v4.2s, v2.2s, v0.2s
	adrp	x7, .LCPI0_3
	adrp	x6, .LCPI0_4
	adrp	x5, .LCPI0_7
	ldr	d1, [x7, :lo12:.LCPI0_3]
	add	v2.2s, v6.2s, v3.2s
	add	v3.2s, v5.2s, v10.2s
	add	v0.2s, v9.2s, v11.2s
	ldr	d9, [x2, #2376]
	ldr	d10, [x1, #3168]
	zip1	v5.2s, v8.2s, v7.2s
	zip2	v7.2s, v8.2s, v7.2s
	adrp	x2, .LCPI0_5
	ldr	d6, [x6, :lo12:.LCPI0_4]
	zip1	v8.2s, v9.2s, v10.2s
	zip2	v9.2s, v9.2s, v10.2s
	ldr	d10, [x2, :lo12:.LCPI0_5]
	adrp	x2, .LCPI0_6
	ldr	d12, [x5, :lo12:.LCPI0_7]
	add	v4.2s, v4.2s, v1.2s
	ldr	d11, [x2, :lo12:.LCPI0_6]
	mul	v1.2s, v3.2s, v2.2s
	add	v2.2s, v5.2s, v6.2s
	ldp	w3, w4, [x1]
	add	v3.2s, v7.2s, v10.2s
	add	v5.2s, v8.2s, v11.2s
	add	v6.2s, v9.2s, v12.2s
	mov	x0, x1
	mul	v0.2s, v4.2s, v0.2s
	madd	w2, w3, w4, w3
	ldr	w3, [x1, #3564]
	ldr	w1, [x1, #3568]
	mul	v3.2s, v3.2s, v2.2s
	mul	v2.2s, v6.2s, v5.2s
	stur	d1, [x0, #4]
	add	w3, w3, #891
	add	w1, w1, #892
	mul	w1, w1, w3
	ldp	d9, d8, [sp, #32]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #16]             // 16-byte Folded Reload
	str	w2, [x0]
	stur	d0, [x0, #12]
	stur	d3, [x0, #20]
	stur	d2, [x0, #28]
	mov	w0, w2
	str	w1, [x0, #36]
	ldr	d12, [sp], #48                  // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
	ret
.Lfunc_end0:
	.size	massive_vreg_test, .Lfunc_end0-massive_vreg_test
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function main
.LCPI1_0:
	.word	0                               // 0x0
	.word	1                               // 0x1
	.word	2                               // 0x2
	.word	3                               // 0x3
	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:                               // %entry
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	str	x28, [sp, #16]                  // 8-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w28, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	sub	sp, sp, #4000
	mov	x0, sp
	mov	w1, wzr
	mov	w2, #4000                       // =0xfa0
	mov	x0, sp
	bl	memset
	movi	v0.4s, #4
	movi	v1.4s, #8
	adrp	x1, .LCPI1_0
	ldr	q2, [x1, :lo12:.LCPI1_0]
	mov	x1, xzr
.LBB1_1:                                // %vector.body
                                        // =>This Inner Loop Header: Depth=1
	add	v3.4s, v2.4s, v0.4s
	add	x2, x0, x1
	add	x1, x1, #32
	cmp	x1, #4000
	stp	q2, q3, [x2]
	add	v2.4s, v2.4s, v1.4s
	b.ne	.LBB1_1
// %bb.2:                               // %for.cond.cleanup
	ldp	w0, w1, [sp]
	madd	w0, w0, w1, w0
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, w0
	bl	printf
	mov	w0, wzr
	add	sp, sp, #4000
	.cfi_def_cfa wsp, 32
	ldr	x28, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w28
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Result: %d\n"
	.size	.L.str, 12

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git 9f790e9e900f8dab0e35b49a5844c2900865231e)"
	.section	".note.GNU-stack","",@progbits
