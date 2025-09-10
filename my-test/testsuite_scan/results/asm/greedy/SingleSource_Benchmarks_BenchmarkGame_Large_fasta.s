	.file	"fasta.c"
	.text
	.globl	main                            // -- Begin function main
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
	movi	d0, #0000000000000000
	adrp	x8, .L_MergedGlobals+8
	add	x8, x8, :lo12:.L_MergedGlobals+8
	ldr	s1, [x8, #32]
	ldr	s2, [x8, #40]
	ldr	s3, [x8, #48]
	ldr	s4, [x8, #56]
	ldr	s5, [x8, #64]
	ldr	s6, [x8, #72]
	ldr	s7, [x8, #80]
	ldr	s16, [x8, #88]
	adrp	x21, :got:stdout
	fadd	s1, s1, s0
	ldr	s17, [x8, #96]
	ldr	s18, [x8, #104]
	ldr	s19, [x8, #112]
	ldr	s20, [x8]
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	mov	w1, #22                         // =0x16
	mov	w2, #1                          // =0x1
	fadd	s0, s20, s0
	ldr	s20, [x8, #120]
	fadd	s2, s1, s2
	str	s1, [x8, #32]
	str	s0, [x8]
	fadd	s3, s2, s3
	str	s2, [x8, #40]
	ldr	s2, [x8, #8]
	fadd	s0, s0, s2
	fadd	s4, s3, s4
	str	s3, [x8, #48]
	ldr	s3, [x8, #128]
	str	s0, [x8, #8]
	fadd	s5, s4, s5
	str	s4, [x8, #56]
	ldr	s4, [x8, #16]
	fadd	s0, s0, s4
	ldr	s4, [x8, #144]
	fadd	s6, s5, s6
	str	s5, [x8, #64]
	ldr	s5, [x8, #24]
	str	s0, [x8, #16]
	fadd	s0, s0, s5
	fadd	s7, s6, s7
	str	s6, [x8, #72]
	fadd	s16, s7, s16
	str	s7, [x8, #80]
	fadd	s17, s16, s17
	str	s16, [x8, #88]
	fadd	s18, s17, s18
	str	s17, [x8, #96]
	fadd	s19, s18, s19
	str	s18, [x8, #104]
	fadd	s1, s19, s20
	str	s19, [x8, #112]
	fadd	s2, s1, s3
	ldr	s3, [x8, #136]
	str	s1, [x8, #120]
	fadd	s3, s2, s3
	str	s2, [x8, #128]
	ldr	x21, [x21, :got_lo12:stdout]
	str	s0, [x8, #24]
	ldr	x3, [x21]
	fadd	s4, s3, s4
	str	s3, [x8, #136]
	str	s4, [x8, #144]
	bl	fwrite
	mov	w0, #347                        // =0x15b
	bl	malloc
	adrp	x20, .L.str
	add	x20, x20, :lo12:.L.str
	mov	w2, #287                        // =0x11f
	mov	x1, x20
	mov	x19, x0
	bl	memcpy
	ldp	q0, q1, [x20]
	add	x8, x19, #287
	ldr	q2, [x20, #32]
	mov	w23, #38528                     // =0x9680
	mov	x22, xzr
	movk	w23, #152, lsl #16
	mov	w24, #60                        // =0x3c
	stp	q0, q1, [x8]
	ldur	q0, [x20, #44]
	str	q2, [x8, #32]
	stur	q0, [x8, #44]
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	cmp	x23, #60
	ldr	x3, [x21]
	add	x0, x19, x22
	csel	x20, x23, x24, lo
	mov	w1, #1                          // =0x1
	mov	x2, x20
	bl	fwrite
	ldr	x1, [x21]
	mov	w0, #10                         // =0xa
	bl	putc
	add	x8, x20, x22
	sub	x9, x8, #287
	cmp	x8, #286
	csel	x22, x9, x8, hi
	subs	x23, x23, x20
	b.ne	.LBB0_1
// %bb.2:
	mov	x0, x19
	bl	free
	ldr	x3, [x21]
	adrp	x0, .L.str.2
	add	x0, x0, :lo12:.L.str.2
	mov	w1, #25                         // =0x19
	mov	w2, #1                          // =0x1
	bl	fwrite
	mov	w8, #45056                      // =0xb000
	mov	x20, #48785                     // =0xbe91
	mov	w22, #57792                     // =0xe1c0
	movk	w8, #18440, lsl #16
	movk	x20, #64876, lsl #16
	mov	w26, #8896                      // =0x22c0
	fmov	s8, w8
	movk	x20, #44289, lsl #32
	movk	w22, #228, lsl #16
	adrp	x19, .L_MergedGlobals
	mov	w24, #3877                      // =0xf25
	mov	w25, #29573                     // =0x7385
	movk	x20, #15342, lsl #48
	movk	w26, #2, lsl #16
	adrp	x27, .L_MergedGlobals+40
	add	x27, x27, :lo12:.L_MergedGlobals+40
	add	x28, sp, #3
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
                                        //       Child Loop BB0_5 Depth 3
	cmp	x22, #60
	ldr	x9, [x19, :lo12:.L_MergedGlobals]
	mov	w10, #60                        // =0x3c
	mov	x8, xzr
	csel	x23, x22, x10, lo
.LBB0_4:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_5 Depth 3
	madd	x9, x9, x24, x25
	umulh	x10, x9, x20
	lsr	x10, x10, #15
	msub	x9, x10, x26, x9
	mov	x10, x27
	ucvtf	s0, x9
	fdiv	s0, s0, s8
.LBB0_5:                                //   Parent Loop BB0_3 Depth=1
                                        //     Parent Loop BB0_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	s1, [x10], #8
	fcmp	s1, s0
	b.mi	.LBB0_5
// %bb.6:                               //   in Loop: Header=BB0_4 Depth=2
	add	x11, x8, #1
	ldurb	w10, [x10, #-4]
	cmp	x11, x23
	strb	w10, [x28, x8]
	mov	x8, x11
	b.ne	.LBB0_4
// %bb.7:                               //   in Loop: Header=BB0_3 Depth=1
	ldr	x3, [x21]
	add	x0, sp, #3
	add	x2, x23, #1
	mov	w1, #1                          // =0x1
	str	x9, [x19, :lo12:.L_MergedGlobals]
	mov	w8, #10                         // =0xa
	strb	w8, [x28, x23]
	bl	fwrite
	subs	x22, x22, x23
	b.ne	.LBB0_3
// %bb.8:
	ldr	x3, [x21]
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	mov	w1, #30                         // =0x1e
	mov	w2, #1                          // =0x1
	bl	fwrite
	mov	w8, #45056                      // =0xb000
	mov	w22, #30784                     // =0x7840
	mov	w26, #8896                      // =0x22c0
	movk	w8, #18440, lsl #16
	movk	w22, #381, lsl #16
	mov	w24, #3877                      // =0xf25
	fmov	s8, w8
	mov	w25, #29573                     // =0x7385
	movk	w26, #2, lsl #16
	adrp	x27, .L_MergedGlobals+8
	add	x27, x27, :lo12:.L_MergedGlobals+8
	add	x28, sp, #3
.LBB0_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_10 Depth 2
                                        //       Child Loop BB0_11 Depth 3
	cmp	x22, #60
	ldr	x9, [x19, :lo12:.L_MergedGlobals]
	mov	w10, #60                        // =0x3c
	mov	x8, xzr
	csel	x23, x22, x10, lo
.LBB0_10:                               //   Parent Loop BB0_9 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_11 Depth 3
	madd	x9, x9, x24, x25
	umulh	x10, x9, x20
	lsr	x10, x10, #15
	msub	x9, x10, x26, x9
	mov	x10, x27
	ucvtf	s0, x9
	fdiv	s0, s0, s8
.LBB0_11:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	s1, [x10], #8
	fcmp	s1, s0
	b.mi	.LBB0_11
// %bb.12:                              //   in Loop: Header=BB0_10 Depth=2
	add	x11, x8, #1
	ldurb	w10, [x10, #-4]
	cmp	x11, x23
	strb	w10, [x28, x8]
	mov	x8, x11
	b.ne	.LBB0_10
// %bb.13:                              //   in Loop: Header=BB0_9 Depth=1
	ldr	x3, [x21]
	add	x0, sp, #3
	add	x2, x23, #1
	mov	w1, #1                          // =0x1
	str	x9, [x19, :lo12:.L_MergedGlobals]
	mov	w8, #10                         // =0xa
	strb	w8, [x28, x23]
	bl	fwrite
	subs	x22, x22, x23
	b.ne	.LBB0_9
// %bb.14:
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
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
	.cfi_endproc
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGACCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAATACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGGAGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
	.size	.L.str, 288

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	">ONE Homo sapiens alu\n"
	.size	.L.str.1, 23

	.type	.L.str.2,@object                // @.str.2
.L.str.2:
	.asciz	">TWO IUB ambiguity codes\n"
	.size	.L.str.2, 26

	.type	.L.str.3,@object                // @.str.3
.L.str.3:
	.asciz	">THREE Homo sapiens frequency\n"
	.size	.L.str.3, 31

	.type	.L_MergedGlobals,@object        // @_MergedGlobals
	.data
	.p2align	3, 0x0
.L_MergedGlobals:
	.xword	42                              // 0x2a
	.word	0x3e9b1ce9                      // float 0.302954942
	.byte	97                              // 0x61
	.zero	3
	.word	0x3e4abd72                      // float 0.197988302
	.byte	99                              // 0x63
	.zero	3
	.word	0x3e4a49d7                      // float 0.197547302
	.byte	103                             // 0x67
	.zero	3
	.word	0x3e9a5f72                      // float 0.30150944
	.byte	116                             // 0x74
	.zero	3
	.word	0x3e8a3d71                      // float 0.270000011
	.byte	97                              // 0x61
	.zero	3
	.word	0x3df5c28f                      // float 0.119999997
	.byte	99                              // 0x63
	.zero	3
	.word	0x3df5c28f                      // float 0.119999997
	.byte	103                             // 0x67
	.zero	3
	.word	0x3e8a3d71                      // float 0.270000011
	.byte	116                             // 0x74
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	66                              // 0x42
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	68                              // 0x44
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	72                              // 0x48
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	75                              // 0x4b
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	77                              // 0x4d
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	78                              // 0x4e
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	82                              // 0x52
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	83                              // 0x53
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	86                              // 0x56
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	87                              // 0x57
	.zero	3
	.word	0x3ca3d70a                      // float 0.0199999996
	.byte	89                              // 0x59
	.zero	3
	.size	.L_MergedGlobals, 160

myrandom.last = .L_MergedGlobals
	.size	myrandom.last, 8
main.homosapiens = .L_MergedGlobals+8
	.size	main.homosapiens, 32
main.iub = .L_MergedGlobals+40
	.size	main.iub, 120
	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
