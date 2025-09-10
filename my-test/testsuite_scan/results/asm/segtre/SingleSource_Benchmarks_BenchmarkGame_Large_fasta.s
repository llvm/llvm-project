	.file	"fasta.c"
	.text
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	str	d12, [sp, #64]                  // 8-byte Folded Spill
	stp	d11, d10, [sp, #80]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #96]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #112]            // 16-byte Folded Spill
	add	x29, sp, #112
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_offset b8, -24
	.cfi_offset b9, -32
	.cfi_offset b10, -40
	.cfi_offset b11, -48
	.cfi_offset b12, -64
	movi	d0, #0000000000000000
	adrp	x1, .L_MergedGlobals+8
	add	x1, x1, :lo12:.L_MergedGlobals+8
	ldr	s1, [x1, #32]
	ldr	s2, [x1, #40]
	ldr	s3, [x1, #48]
	ldr	s4, [x1, #56]
	ldr	s5, [x1, #64]
	ldr	s6, [x1, #72]
	ldr	s7, [x1, #80]
	ldr	s8, [x1, #88]
	adrp	x0, :got:stdout
	fadd	s1, s1, s0
	ldr	s9, [x1, #96]
	ldr	s10, [x1, #104]
	ldr	s11, [x1, #112]
	ldr	s12, [x1]
	fadd	s12, s12, s0
	ldr	s0, [x1, #120]
	fadd	s2, s1, s2
	str	s1, [x1, #32]
	ldr	s1, [x1, #128]
	str	s12, [x1]
	fadd	s3, s2, s3
	str	s2, [x1, #40]
	ldr	s2, [x1, #8]
	fadd	s2, s12, s2
	fadd	s4, s3, s4
	str	s3, [x1, #48]
	ldr	s3, [x1, #16]
	fadd	s3, s2, s3
	str	s2, [x1, #8]
	fadd	s5, s4, s5
	str	s4, [x1, #56]
	ldr	s4, [x1, #136]
	str	s3, [x1, #16]
	fadd	s6, s5, s6
	str	s5, [x1, #64]
	ldr	s5, [x1, #24]
	fadd	s3, s3, s5
	fadd	s7, s6, s7
	str	s6, [x1, #72]
	fadd	s8, s7, s8
	str	s7, [x1, #80]
	fadd	s9, s8, s9
	str	s8, [x1, #88]
	fadd	s10, s9, s10
	str	s9, [x1, #96]
	fadd	s11, s10, s11
	str	s10, [x1, #104]
	fadd	s0, s11, s0
	str	s11, [x1, #112]
	fadd	s1, s0, s1
	str	s0, [x1, #120]
	fadd	s2, s1, s4
	ldr	s4, [x1, #144]
	str	s1, [x1, #128]
	ldr	x0, [x0, :got_lo12:stdout]
	str	s3, [x1, #24]
	ldr	x2, [x0]
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	fadd	s4, s2, s4
	mov	w2, #1                          // =0x1
	str	s2, [x1, #136]
	mov	x3, x2
	str	s4, [x1, #144]
	mov	w1, #22                         // =0x16
	bl	fwrite
	mov	w0, #347                        // =0x15b
	bl	malloc
	adrp	x3, .L.str
	add	x3, x3, :lo12:.L.str
	mov	w2, #287                        // =0x11f
	mov	x1, x3
	bl	memcpy
	ldp	q0, q1, [x3]
	add	x4, x1, #287
	ldr	q2, [x3, #32]
	mov	x2, xzr
	stp	q0, q1, [x4]
	ldur	q0, [x3, #44]
	mov	w3, #38528                      // =0x9680
	str	q2, [x4, #32]
	movk	w3, #152, lsl #16
	stur	q0, [x4, #44]
	mov	w4, #60                         // =0x3c
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
	cmp	x3, #60
	ldr	x6, [x0]
	add	x0, x1, x2
	csel	x5, x3, x4, lo
	mov	w1, #1                          // =0x1
	mov	x2, x5
	mov	x3, x6
	bl	fwrite
	ldr	x1, [x0]
	mov	w0, #10                         // =0xa
	bl	putc
	add	x2, x5, x2
	sub	x6, x2, #287
	cmp	x2, #286
	csel	x2, x6, x2, hi
	subs	x3, x3, x5
	b.ne	.LBB0_1
// %bb.2:
	mov	x0, x1
	bl	free
	ldr	x1, [x0]
	adrp	x0, .L.str.2
	add	x0, x0, :lo12:.L.str.2
	mov	w1, #25                         // =0x19
	mov	w2, #1                          // =0x1
	mov	x3, x1
	bl	fwrite
	mov	w9, #45056                      // =0xb000
	mov	x2, #48785                      // =0xbe91
	mov	w3, #57792                      // =0xe1c0
	movk	w9, #18440, lsl #16
	movk	x2, #64876, lsl #16
	mov	w7, #8896                       // =0x22c0
	fmov	s0, w9
	movk	x2, #44289, lsl #32
	movk	w3, #228, lsl #16
	mov	w4, #60                         // =0x3c
	adrp	x1, .L_MergedGlobals
	mov	w5, #3877                       // =0xf25
	mov	w6, #29573                      // =0x7385
	movk	x2, #15342, lsl #48
	movk	w7, #2, lsl #16
	adrp	x8, .L_MergedGlobals+40
	add	x8, x8, :lo12:.L_MergedGlobals+40
	add	x9, sp, #3
	mov	w10, #10                        // =0xa
.LBB0_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_4 Depth 2
                                        //       Child Loop BB0_5 Depth 3
	cmp	x3, #60
	ldr	x13, [x1, :lo12:.L_MergedGlobals]
	mov	x12, xzr
	csel	x11, x3, x4, lo
.LBB0_4:                                //   Parent Loop BB0_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_5 Depth 3
	madd	x13, x13, x5, x6
	umulh	x14, x13, x2
	lsr	x14, x14, #15
	msub	x13, x14, x7, x13
	mov	x14, x8
	ucvtf	s1, x13
	fdiv	s1, s1, s0
.LBB0_5:                                //   Parent Loop BB0_3 Depth=1
                                        //     Parent Loop BB0_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	s2, [x14], #8
	fcmp	s2, s1
	b.mi	.LBB0_5
// %bb.6:                               //   in Loop: Header=BB0_4 Depth=2
	add	x15, x12, #1
	ldurb	w14, [x14, #-4]
	cmp	x15, x11
	strb	w14, [x9, x12]
	mov	x12, x15
	b.ne	.LBB0_4
// %bb.7:                               //   in Loop: Header=BB0_3 Depth=1
	ldr	x3, [x0]
	str	x13, [x1, :lo12:.L_MergedGlobals]
	add	x0, sp, #3
	add	x2, x11, #1
	mov	w1, #1                          // =0x1
	strb	w10, [x9, x11]
	bl	fwrite
	subs	x3, x3, x11
	b.ne	.LBB0_3
// %bb.8:
	ldr	x3, [x0]
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	mov	w1, #30                         // =0x1e
	mov	w2, #1                          // =0x1
	bl	fwrite
	mov	w9, #45056                      // =0xb000
	mov	w3, #30784                      // =0x7840
	mov	w7, #8896                       // =0x22c0
	movk	w9, #18440, lsl #16
	movk	w3, #381, lsl #16
	mov	w4, #60                         // =0x3c
	fmov	s0, w9
	mov	w5, #3877                       // =0xf25
	mov	w6, #29573                      // =0x7385
	movk	w7, #2, lsl #16
	adrp	x8, .L_MergedGlobals+8
	add	x8, x8, :lo12:.L_MergedGlobals+8
	add	x9, sp, #3
	mov	w10, #10                        // =0xa
.LBB0_9:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_10 Depth 2
                                        //       Child Loop BB0_11 Depth 3
	cmp	x3, #60
	ldr	x13, [x1, :lo12:.L_MergedGlobals]
	mov	x12, xzr
	csel	x11, x3, x4, lo
.LBB0_10:                               //   Parent Loop BB0_9 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB0_11 Depth 3
	madd	x13, x13, x5, x6
	umulh	x14, x13, x2
	lsr	x14, x14, #15
	msub	x13, x14, x7, x13
	mov	x14, x8
	ucvtf	s1, x13
	fdiv	s1, s1, s0
.LBB0_11:                               //   Parent Loop BB0_9 Depth=1
                                        //     Parent Loop BB0_10 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	s2, [x14], #8
	fcmp	s2, s1
	b.mi	.LBB0_11
// %bb.12:                              //   in Loop: Header=BB0_10 Depth=2
	add	x15, x12, #1
	ldurb	w14, [x14, #-4]
	cmp	x15, x11
	strb	w14, [x9, x12]
	mov	x12, x15
	b.ne	.LBB0_10
// %bb.13:                              //   in Loop: Header=BB0_9 Depth=1
	ldr	x3, [x0]
	str	x13, [x1, :lo12:.L_MergedGlobals]
	add	x0, sp, #3
	add	x2, x11, #1
	mov	w1, #1                          // =0x1
	strb	w10, [x9, x11]
	bl	fwrite
	subs	x3, x3, x11
	b.ne	.LBB0_9
// %bb.14:
	mov	w0, wzr
	.cfi_def_cfa wsp, 128
	ldp	x29, x30, [sp, #112]            // 16-byte Folded Reload
	ldr	d12, [sp, #64]                  // 8-byte Folded Reload
	ldp	d9, d8, [sp, #96]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #80]             // 16-byte Folded Reload
	add	sp, sp, #128
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	.cfi_restore b9
	.cfi_restore b10
	.cfi_restore b11
	.cfi_restore b12
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
