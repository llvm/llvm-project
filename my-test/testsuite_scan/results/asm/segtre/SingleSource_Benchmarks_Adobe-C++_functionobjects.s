	.file	"functionobjects.cpp"
	.text
	.globl	_Z13record_resultdPKc           // -- Begin function _Z13record_resultdPKc
	.p2align	2
	.type	_Z13record_resultdPKc,@function
_Z13record_resultdPKc:                  // @_Z13record_resultdPKc
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	adrp	x4, results
	adrp	x3, allocated_results
	adrp	x1, current_test
	ldr	x2, [x4, :lo12:results]
	ldr	w6, [x3, :lo12:allocated_results]
	cbz	x2, .LBB0_2
// %bb.1:
	ldr	w5, [x1, :lo12:current_test]
	cmp	w5, w6
	b.lt	.LBB0_4
.LBB0_2:
	add	w5, w6, #10
	mov	x0, x2
	sbfiz	x1, x5, #4, #32
	str	w5, [x3, :lo12:allocated_results]
	bl	realloc
	str	x0, [x4, :lo12:results]
	cbz	x0, .LBB0_5
// %bb.3:
	ldr	w5, [x1, :lo12:current_test]
	mov	x2, x0
.LBB0_4:
	add	x2, x2, w5, sxtw #4
	add	w3, w5, #1
	str	d0, [x2]
	str	x0, [x2, #8]
	str	w3, [x1, :lo12:current_test]
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB0_5:
	.cfi_restore_state
	ldr	w0, [x3, :lo12:allocated_results]
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
	mov	w1, w0
	bl	printf
	mov	w0, #-1                         // =0xffffffff
	bl	exit
.Lfunc_end0:
	.size	_Z13record_resultdPKc, .Lfunc_end0-_Z13record_resultdPKc
	.cfi_endproc
                                        // -- End function
	.globl	_Z9summarizePKciiii             // -- Begin function _Z9summarizePKciiii
	.p2align	2
	.type	_Z9summarizePKciiii,@function
_Z9summarizePKciiii:                    // @_Z9summarizePKciiii
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	w6, w1
	mov	x1, x0
	adrp	x0, current_test
	ldr	w7, [x0, :lo12:current_test]
	mov	w2, w4
	adrp	x3, results
	mov	w5, w2
	cmp	w7, #1
	b.lt	.LBB1_3
// %bb.1:
	ldr	x4, [x3, :lo12:results]
	add	x8, x4, #8
	mov	w4, #12                         // =0xc
.LBB1_2:                                // =>This Inner Loop Header: Depth=1
	ldr	x0, [x8], #16
	bl	strlen
	mov	x9, x0
	cmp	w4, w9
	csel	w4, w4, w9, gt
	subs	x7, x7, #1
	b.ne	.LBB1_2
	b	.LBB1_4
.LBB1_3:
	mov	w4, #12                         // =0xc
.LBB1_4:
	adrp	x7, .L.str.2
	add	x7, x7, :lo12:.L.str.2
	sub	w1, w4, #12
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	mov	x2, x7
	bl	printf
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	mov	w1, w4
	mov	x2, x7
	bl	printf
	movi	d0, #0000000000000000
	ldr	w7, [x0, :lo12:current_test]
	cmp	w7, #1
	b.lt	.LBB1_15
// %bb.5:                               // %.preheader2
	scvtf	d1, w6
	scvtf	d2, w5
	mov	x5, #145685290680320            // =0x848000000000
	movk	x5, #16686, lsl #48
	mov	x7, xzr
	mov	x8, xzr
	adrp	x6, .L.str.4
	add	x6, x6, :lo12:.L.str.4
	adrp	x9, .L.str.5
	add	x9, x9, :lo12:.L.str.5
	fmul	d1, d1, d2
	fmov	d2, x5
	fdiv	d1, d1, d2
.LBB1_6:                                // =>This Inner Loop Header: Depth=1
	ldr	x10, [x3, :lo12:results]
	add	x11, x10, x7
	ldr	x5, [x11, #8]
	mov	x0, x5
	bl	strlen
	ldr	d0, [x11]
	ldr	d4, [x10]
	mov	x10, x0
	sub	w10, w4, w10
	mov	x0, x6
	mov	w1, w8
	fdiv	d1, d1, d0
	mov	w2, w10
	mov	x3, x9
	mov	x4, x5
	fdiv	d2, d0, d4
	bl	printf
	ldrsw	x5, [x0, :lo12:current_test]
	add	x8, x8, #1
	add	x7, x7, #16
	cmp	x8, x5
	b.lt	.LBB1_6
// %bb.7:
	cmp	w5, #1
	b.lt	.LBB1_15
// %bb.8:
	movi	d0, #0000000000000000
	ldr	x4, [x3, :lo12:results]
	cmp	w5, #1
	b.ne	.LBB1_10
// %bb.9:
	mov	x6, xzr
	b	.LBB1_13
.LBB1_10:
	and	x6, x5, #0x7ffffffe
	add	x7, x4, #16
	mov	x8, x6
.LBB1_11:                               // =>This Inner Loop Header: Depth=1
	ldur	d1, [x7, #-16]
	subs	x8, x8, #2
	fadd	d0, d0, d1
	ldr	d1, [x7], #32
	fadd	d0, d0, d1
	b.ne	.LBB1_11
// %bb.12:
	cmp	x6, x5
	b.eq	.LBB1_15
.LBB1_13:                               // %.preheader
	add	x4, x4, x6, lsl #4
	sub	x5, x5, x6
.LBB1_14:                               // =>This Inner Loop Header: Depth=1
	ldr	d1, [x4], #16
	subs	x5, x5, #1
	fadd	d0, d0, d1
	b.ne	.LBB1_14
.LBB1_15:
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	bl	printf
	cbz	w2, .LBB1_20
// %bb.16:
	ldr	w2, [x0, :lo12:current_test]
	cmp	w2, #2
	b.lt	.LBB1_20
// %bb.17:
	ldr	x2, [x3, :lo12:results]
	movi	d0, #0000000000000000
	mov	w3, #1                          // =0x1
	ldr	d1, [x2], #16
.LBB1_18:                               // =>This Inner Loop Header: Depth=1
	ldr	d2, [x2], #16
	fdiv	d0, d2, d1
	bl	log
	fadd	d0, d0, d0
	ldrsw	x4, [x0, :lo12:current_test]
	add	x3, x3, #1
	cmp	x3, x4
	b.lt	.LBB1_18
// %bb.19:
	sub	w2, w4, #1
	scvtf	d1, w2
	fdiv	d0, d0, d1
	bl	exp
	adrp	x0, .L.str.7
	add	x0, x0, :lo12:.L.str.7
	bl	printf
.LBB1_20:
	str	wzr, [x0, :lo12:current_test]
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end1:
	.size	_Z9summarizePKciiii, .Lfunc_end1-_Z9summarizePKciiii
	.cfi_endproc
                                        // -- End function
	.globl	_Z17summarize_simplefP8_IO_FILEPKc // -- Begin function _Z17summarize_simplefP8_IO_FILEPKc
	.p2align	2
	.type	_Z17summarize_simplefP8_IO_FILEPKc,@function
_Z17summarize_simplefP8_IO_FILEPKc:     // @_Z17summarize_simplefP8_IO_FILEPKc
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x2, x0
	adrp	x0, current_test
	adrp	x3, results
	ldr	w5, [x0, :lo12:current_test]
	cmp	w5, #1
	b.lt	.LBB2_3
// %bb.1:
	ldr	x4, [x3, :lo12:results]
	add	x6, x4, #8
	mov	w4, #12                         // =0xc
.LBB2_2:                                // =>This Inner Loop Header: Depth=1
	ldr	x0, [x6], #16
	bl	strlen
	mov	x7, x0
	cmp	w4, w7
	csel	w4, w4, w7, gt
	subs	x5, x5, #1
	b.ne	.LBB2_2
	b	.LBB2_4
.LBB2_3:
	mov	w4, #12                         // =0xc
.LBB2_4:
	sub	w6, w4, #12
	adrp	x5, .L.str.2
	add	x5, x5, :lo12:.L.str.2
	adrp	x1, .L.str.8
	add	x1, x1, :lo12:.L.str.8
	mov	x0, x2
	mov	w2, w6
	mov	x3, x5
	bl	fprintf
	adrp	x1, .L.str.9
	add	x1, x1, :lo12:.L.str.9
	mov	x0, x2
	mov	w2, w4
	mov	x3, x5
	bl	fprintf
	movi	d0, #0000000000000000
	ldr	w5, [x0, :lo12:current_test]
	cmp	w5, #1
	b.lt	.LBB2_15
// %bb.5:                               // %.preheader2
	mov	x6, xzr
	mov	x7, xzr
	adrp	x8, .L.str.10
	add	x8, x8, :lo12:.L.str.10
	adrp	x9, .L.str.5
	add	x9, x9, :lo12:.L.str.5
.LBB2_6:                                // =>This Inner Loop Header: Depth=1
	ldr	x5, [x3, :lo12:results]
	add	x10, x5, x6
	ldr	x5, [x10, #8]
	mov	x0, x5
	bl	strlen
	mov	x11, x0
	ldr	d0, [x10]
	mov	x0, x2
	sub	w3, w4, w11
	mov	x1, x8
	mov	w2, w7
	mov	x4, x9
	bl	fprintf
	ldrsw	x5, [x0, :lo12:current_test]
	add	x7, x7, #1
	add	x6, x6, #16
	cmp	x7, x5
	b.lt	.LBB2_6
// %bb.7:
	cmp	w5, #1
	b.lt	.LBB2_15
// %bb.8:
	movi	d0, #0000000000000000
	ldr	x3, [x3, :lo12:results]
	cmp	w5, #1
	b.ne	.LBB2_10
// %bb.9:
	mov	x4, xzr
	b	.LBB2_13
.LBB2_10:
	and	x4, x5, #0x7ffffffe
	add	x6, x3, #16
	mov	x7, x4
.LBB2_11:                               // =>This Inner Loop Header: Depth=1
	ldur	d1, [x6, #-16]
	subs	x7, x7, #2
	fadd	d0, d0, d1
	ldr	d1, [x6], #32
	fadd	d0, d0, d1
	b.ne	.LBB2_11
// %bb.12:
	cmp	x4, x5
	b.eq	.LBB2_15
.LBB2_13:                               // %.preheader
	add	x3, x3, x4, lsl #4
	sub	x4, x5, x4
.LBB2_14:                               // =>This Inner Loop Header: Depth=1
	ldr	d1, [x3], #16
	subs	x4, x4, #1
	fadd	d0, d0, d1
	b.ne	.LBB2_14
.LBB2_15:
	adrp	x1, .L.str.6
	add	x1, x1, :lo12:.L.str.6
	mov	x0, x2
	mov	x2, x1
	bl	fprintf
	str	wzr, [x0, :lo12:current_test]
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end2:
	.size	_Z17summarize_simplefP8_IO_FILEPKc, .Lfunc_end2-_Z17summarize_simplefP8_IO_FILEPKc
	.cfi_endproc
                                        // -- End function
	.globl	_Z11start_timerv                // -- Begin function _Z11start_timerv
	.p2align	2
	.type	_Z11start_timerv,@function
_Z11start_timerv:                       // @_Z11start_timerv
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	clock
	adrp	x1, start_time
	str	x0, [x1, :lo12:start_time]
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end3:
	.size	_Z11start_timerv, .Lfunc_end3-_Z11start_timerv
	.cfi_endproc
                                        // -- End function
	.globl	_Z5timerv                       // -- Begin function _Z5timerv
	.p2align	2
	.type	_Z5timerv,@function
_Z5timerv:                              // @_Z5timerv
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	bl	clock
	adrp	x0, start_time
	ldr	x1, [x0, :lo12:start_time]
	sub	x1, x0, x1
	scvtf	d0, x1
	mov	x1, #145685290680320            // =0x848000000000
	movk	x1, #16686, lsl #48
	fmov	d1, x1
	adrp	x1, end_time
	fdiv	d0, d0, d1
	str	x0, [x1, :lo12:end_time]
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end4:
	.size	_Z5timerv, .Lfunc_end4-_Z5timerv
	.cfi_endproc
                                        // -- End function
	.globl	_Z19less_than_function1PKvS0_   // -- Begin function _Z19less_than_function1PKvS0_
	.p2align	2
	.type	_Z19less_than_function1PKvS0_,@function
_Z19less_than_function1PKvS0_:          // @_Z19less_than_function1PKvS0_
	.cfi_startproc
// %bb.0:
	ldr	d0, [x1]
	ldr	d1, [x1]
	fcmp	d0, d1
	cset	w0, gt
	csinv	w0, w0, wzr, pl
	ret
.Lfunc_end5:
	.size	_Z19less_than_function1PKvS0_, .Lfunc_end5-_Z19less_than_function1PKvS0_
	.cfi_endproc
                                        // -- End function
	.globl	_Z19less_than_function2dd       // -- Begin function _Z19less_than_function2dd
	.p2align	2
	.type	_Z19less_than_function2dd,@function
_Z19less_than_function2dd:              // @_Z19less_than_function2dd
	.cfi_startproc
// %bb.0:
	fcmp	d1, d1
	cset	w0, mi
	ret
.Lfunc_end6:
	.size	_Z19less_than_function2dd, .Lfunc_end6-_Z19less_than_function2dd
	.cfi_endproc
                                        // -- End function
	.globl	_ZNK17less_than_functorclERKdS1_ // -- Begin function _ZNK17less_than_functorclERKdS1_
	.p2align	2
	.type	_ZNK17less_than_functorclERKdS1_,@function
_ZNK17less_than_functorclERKdS1_:       // @_ZNK17less_than_functorclERKdS1_
	.cfi_startproc
// %bb.0:
	ldr	d0, [x1]
	ldr	d1, [x2]
	fcmp	d0, d1
	cset	w0, mi
	ret
.Lfunc_end7:
	.size	_ZNK17less_than_functorclERKdS1_, .Lfunc_end7-_ZNK17less_than_functorclERKdS1_
	.cfi_endproc
                                        // -- End function
	.globl	_Z18quicksort_functionPdS_PFbddE // -- Begin function _Z18quicksort_functionPdS_PFbddE
	.p2align	2
	.type	_Z18quicksort_functionPdS_PFbddE,@function
_Z18quicksort_functionPdS_PFbddE:       // @_Z18quicksort_functionPdS_PFbddE
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	mov	x0, x1
	sub	x1, x1, x0
	cmp	x1, #9
	b.lt	.LBB8_8
// %bb.1:
	mov	x2, x0
	ldr	d0, [x0]
	mov	x3, x0
	mov	x1, x2
	mov	x4, x0
.LBB8_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_5 Depth 2
	ldr	d1, [x3, #-8]!
	blr	x1
	tbnz	w0, #0, .LBB8_2
// %bb.3:                               //   in Loop: Header=BB8_2 Depth=1
	cmp	x4, x3
	b.hs	.LBB8_9
// %bb.4:                               // %.preheader
                                        //   in Loop: Header=BB8_2 Depth=1
	sub	x4, x4, #8
.LBB8_5:                                //   Parent Loop BB8_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x4, #8]!
	fmov	d1, d0
	blr	x1
	tbnz	w0, #0, .LBB8_5
// %bb.6:                               //   in Loop: Header=BB8_2 Depth=1
	cmp	x4, x3
	b.hs	.LBB8_9
// %bb.7:                               //   in Loop: Header=BB8_2 Depth=1
	ldr	d2, [x4]
	ldr	d1, [x3]
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB8_2
.LBB8_8:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB8_9:
	.cfi_restore_state
	add	x1, x3, #8
	mov	x0, x2
	mov	x2, x1
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	add	x0, x3, #8
	mov	x1, x0
	mov	x2, x0
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	b	_Z9quicksortIPdPFbddEEvT_S3_T0_
.Lfunc_end8:
	.size	_Z18quicksort_functionPdS_PFbddE, .Lfunc_end8-_Z18quicksort_functionPdS_PFbddE
	.cfi_endproc
                                        // -- End function
	.section	.text._Z9quicksortIPdPFbddEEvT_S3_T0_,"axG",@progbits,_Z9quicksortIPdPFbddEEvT_S3_T0_,comdat
	.weak	_Z9quicksortIPdPFbddEEvT_S3_T0_ // -- Begin function _Z9quicksortIPdPFbddEEvT_S3_T0_
	.p2align	2
	.type	_Z9quicksortIPdPFbddEEvT_S3_T0_,@function
_Z9quicksortIPdPFbddEEvT_S3_T0_:        // @_Z9quicksortIPdPFbddEEvT_S3_T0_
	.cfi_startproc
// %bb.0:
	mov	x0, x1
	sub	x1, x1, x0
	cmp	x1, #9
	b.lt	.LBB9_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x3, x0
	mov	x1, x2
	sub	x2, x0, #8
	b	.LBB9_3
.LBB9_2:                                //   in Loop: Header=BB9_3 Depth=1
	add	x4, x4, #8
	mov	x0, x3
	mov	x1, x4
	mov	x2, x4
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	cmp	x6, #8
	mov	x3, x4
	b.le	.LBB9_11
.LBB9_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_4 Depth 2
                                        //       Child Loop BB9_5 Depth 3
                                        //       Child Loop BB9_8 Depth 3
	ldr	d0, [x3]
	mov	x4, x0
	mov	x5, x3
.LBB9_4:                                //   Parent Loop BB9_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB9_5 Depth 3
                                        //       Child Loop BB9_8 Depth 3
	sub	x6, x2, x4
.LBB9_5:                                //   Parent Loop BB9_3 Depth=1
                                        //     Parent Loop BB9_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x4, #-8]!
	blr	x1
	add	x6, x6, #8
	tbnz	w0, #0, .LBB9_5
// %bb.6:                               //   in Loop: Header=BB9_4 Depth=2
	cmp	x5, x4
	b.hs	.LBB9_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB9_4 Depth=2
	sub	x5, x5, #8
.LBB9_8:                                //   Parent Loop BB9_3 Depth=1
                                        //     Parent Loop BB9_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d0, [x5, #8]!
	fmov	d1, d0
	blr	x1
	tbnz	w0, #0, .LBB9_8
// %bb.9:                               //   in Loop: Header=BB9_4 Depth=2
	cmp	x5, x4
	b.hs	.LBB9_2
// %bb.10:                              //   in Loop: Header=BB9_4 Depth=2
	ldr	d2, [x5]
	ldr	d1, [x4]
	str	d2, [x4]
	str	d1, [x5]
	b	.LBB9_4
.LBB9_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB9_12:
	ret
.Lfunc_end9:
	.size	_Z9quicksortIPdPFbddEEvT_S3_T0_, .Lfunc_end9-_Z9quicksortIPdPFbddEEvT_S3_T0_
	.cfi_endproc
                                        // -- End function
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
	cmp	w0, #2
	b.lt	.LBB10_3
// %bb.1:
	ldr	x0, [x1, #8]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	cmp	w0, #2
	b.ne	.LBB10_8
// %bb.2:
	mov	w3, #10000                      // =0x2710
	add	w0, w3, #123
	bl	srand
	sxtw	x1, w3
	sbfiz	x2, x3, #3, #32
	tbz	w3, #31, .LBB10_4
	b	.LBB10_9
.LBB10_3:
	mov	w3, #10000                      // =0x2710
	mov	w0, #300                        // =0x12c
	add	w0, w3, #123
	bl	srand
	sxtw	x1, w3
	sbfiz	x2, x3, #3, #32
	tbnz	w3, #31, .LBB10_9
.LBB10_4:
	mov	x0, x2
	bl	_Znam
	mov	x3, x0
	cbz	w1, .LBB10_7
// %bb.5:
	ubfiz	x4, x1, #3, #32
	mov	x5, xzr
.LBB10_6:                               // =>This Inner Loop Header: Depth=1
	bl	rand
	scvtf	d0, w0
	str	d0, [x3, x5]
	add	x5, x5, #8
	cmp	x4, x5
	b.ne	.LBB10_6
.LBB10_7:
	mov	x4, x2
	mov	x0, x4
	bl	_Znam
	cmp	w0, #1
	mov	x4, x0
	b.ge	.LBB10_10
	b	.LBB10_190
.LBB10_8:
	ldr	x0, [x1, #16]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	mov	x1, x0
	mov	w3, w1
	add	w0, w3, #123
	bl	srand
	sxtw	x1, w3
	sbfiz	x2, x3, #3, #32
	tbz	w3, #31, .LBB10_4
.LBB10_9:
	mov	x0, #-1                         // =0xffffffffffffffff
	mov	x4, #-1                         // =0xffffffffffffffff
	bl	_Znam
	mov	x0, x4
	bl	_Znam
	cmp	w0, #1
	mov	x4, x0
	b.lt	.LBB10_190
.LBB10_10:
	mov	w7, wzr
	add	x8, x4, #8
	sub	x5, x2, #8
	adrp	x9, _Z19less_than_function1PKvS0_
	add	x9, x9, :lo12:_Z19less_than_function1PKvS0_
	adrp	x6, current_test
	adrp	x10, .L.str.11
	add	x10, x10, :lo12:.L.str.11
	b	.LBB10_13
.LBB10_11:                              //   in Loop: Header=BB10_13 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x10
	bl	printf
.LBB10_12:                              //   in Loop: Header=BB10_13 Depth=1
	add	w7, w7, #1
	cmp	w7, w0
	b.eq	.LBB10_20
.LBB10_13:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_16 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_18
// %bb.14:                              //   in Loop: Header=BB10_13 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_15:                              //   in Loop: Header=BB10_13 Depth=1
	mov	x0, x4
	mov	w2, #8                          // =0x8
	mov	x3, x9
	bl	qsort
	mov	x11, x5
	mov	x12, x8
.LBB10_16:                              //   Parent Loop BB10_13 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x11, .LBB10_12
// %bb.17:                              //   in Loop: Header=BB10_16 Depth=2
	ldp	d1, d0, [x12, #-8]
	add	x12, x12, #8
	sub	x11, x11, #8
	fcmp	d0, d1
	b.pl	.LBB10_16
	b	.LBB10_11
.LBB10_18:                              //   in Loop: Header=BB10_13 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_15
// %bb.19:                              //   in Loop: Header=BB10_13 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_15
.LBB10_20:
	add	x7, x4, x1, lsl #3
	mov	w8, wzr
	adrp	x9, _Z19less_than_function2dd
	add	x9, x9, :lo12:_Z19less_than_function2dd
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_23
.LBB10_21:                              //   in Loop: Header=BB10_23 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_22:                              //   in Loop: Header=BB10_23 Depth=1
	add	w8, w8, #1
	cmp	w8, w0
	b.eq	.LBB10_37
.LBB10_23:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_25 Depth 2
                                        //       Child Loop BB10_28 Depth 3
                                        //     Child Loop BB10_33 Depth 2
	cmp	w1, #1
	b.le	.LBB10_35
// %bb.24:                              //   in Loop: Header=BB10_23 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
	ldr	d0, [x4]
	mov	x12, x7
	mov	x13, x4
.LBB10_25:                              //   Parent Loop BB10_23 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_28 Depth 3
	ldr	d1, [x12, #-8]!
	fcmp	d0, d1
	b.mi	.LBB10_25
// %bb.26:                              //   in Loop: Header=BB10_25 Depth=2
	cmp	x13, x12
	b.hs	.LBB10_31
// %bb.27:                              // %.preheader28
                                        //   in Loop: Header=BB10_25 Depth=2
	sub	x13, x13, #8
.LBB10_28:                              //   Parent Loop BB10_23 Depth=1
                                        //     Parent Loop BB10_25 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x13, #8]!
	fcmp	d2, d0
	b.mi	.LBB10_28
// %bb.29:                              //   in Loop: Header=BB10_25 Depth=2
	cmp	x13, x12
	b.hs	.LBB10_31
// %bb.30:                              //   in Loop: Header=BB10_25 Depth=2
	str	d2, [x12]
	str	d1, [x13]
	b	.LBB10_25
.LBB10_31:                              //   in Loop: Header=BB10_23 Depth=1
	add	x1, x12, #8
	mov	x0, x4
	mov	x2, x9
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	add	x0, x12, #8
	mov	x1, x7
	mov	x2, x9
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
.LBB10_32:                              // %.preheader29
                                        //   in Loop: Header=BB10_23 Depth=1
	mov	x12, x5
	mov	x13, x10
.LBB10_33:                              //   Parent Loop BB10_23 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_22
// %bb.34:                              //   in Loop: Header=BB10_33 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_33
	b	.LBB10_21
.LBB10_35:                              //   in Loop: Header=BB10_23 Depth=1
	b.ne	.LBB10_32
// %bb.36:                              //   in Loop: Header=BB10_23 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_32
.LBB10_37:
	mov	w8, wzr
	adrp	x9, _Z19less_than_function2dd
	add	x9, x9, :lo12:_Z19less_than_function2dd
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_40
.LBB10_38:                              //   in Loop: Header=BB10_40 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_39:                              //   in Loop: Header=BB10_40 Depth=1
	add	w8, w8, #1
	cmp	w8, w0
	b.eq	.LBB10_47
.LBB10_40:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_43 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_45
// %bb.41:                              //   in Loop: Header=BB10_40 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_42:                              //   in Loop: Header=BB10_40 Depth=1
	mov	x0, x4
	mov	x1, x7
	mov	x2, x9
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	mov	x12, x5
	mov	x13, x10
.LBB10_43:                              //   Parent Loop BB10_40 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_39
// %bb.44:                              //   in Loop: Header=BB10_43 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_43
	b	.LBB10_38
.LBB10_45:                              //   in Loop: Header=BB10_40 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_42
// %bb.46:                              //   in Loop: Header=BB10_40 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_42
.LBB10_47:
	mov	w8, wzr
	adrp	x9, _Z19less_than_function2dd
	add	x9, x9, :lo12:_Z19less_than_function2dd
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_50
.LBB10_48:                              //   in Loop: Header=BB10_50 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_49:                              //   in Loop: Header=BB10_50 Depth=1
	add	w8, w8, #1
	cmp	w8, w0
	b.eq	.LBB10_64
.LBB10_50:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_52 Depth 2
                                        //       Child Loop BB10_55 Depth 3
                                        //     Child Loop BB10_60 Depth 2
	cmp	w1, #1
	b.le	.LBB10_62
// %bb.51:                              //   in Loop: Header=BB10_50 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
	ldr	d0, [x4]
	mov	x12, x7
	mov	x13, x4
.LBB10_52:                              //   Parent Loop BB10_50 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_55 Depth 3
	ldr	d1, [x12, #-8]!
	fcmp	d0, d1
	b.mi	.LBB10_52
// %bb.53:                              //   in Loop: Header=BB10_52 Depth=2
	cmp	x13, x12
	b.hs	.LBB10_58
// %bb.54:                              // %.preheader24
                                        //   in Loop: Header=BB10_52 Depth=2
	sub	x13, x13, #8
.LBB10_55:                              //   Parent Loop BB10_50 Depth=1
                                        //     Parent Loop BB10_52 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x13, #8]!
	fcmp	d2, d0
	b.mi	.LBB10_55
// %bb.56:                              //   in Loop: Header=BB10_52 Depth=2
	cmp	x13, x12
	b.hs	.LBB10_58
// %bb.57:                              //   in Loop: Header=BB10_52 Depth=2
	str	d2, [x12]
	str	d1, [x13]
	b	.LBB10_52
.LBB10_58:                              //   in Loop: Header=BB10_50 Depth=1
	add	x1, x12, #8
	mov	x0, x4
	mov	x2, x9
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	add	x0, x12, #8
	mov	x1, x7
	mov	x2, x9
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
.LBB10_59:                              // %.preheader25
                                        //   in Loop: Header=BB10_50 Depth=1
	mov	x12, x5
	mov	x13, x10
.LBB10_60:                              //   Parent Loop BB10_50 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_49
// %bb.61:                              //   in Loop: Header=BB10_60 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_60
	b	.LBB10_48
.LBB10_62:                              //   in Loop: Header=BB10_50 Depth=1
	b.ne	.LBB10_59
// %bb.63:                              //   in Loop: Header=BB10_50 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_59
.LBB10_64:
	clz	x8, x1
	mov	w9, wzr
	add	x10, x4, x2
	lsl	x8, x8, #1
	add	x11, x4, #8
	adrp	x12, _Z19less_than_function2dd
	add	x12, x12, :lo12:_Z19less_than_function2dd
	add	x13, x4, #128
	adrp	x14, .L.str.11
	add	x14, x14, :lo12:.L.str.11
	b	.LBB10_67
.LBB10_65:                              //   in Loop: Header=BB10_67 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x14
	bl	printf
.LBB10_66:                              //   in Loop: Header=BB10_67 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_102
.LBB10_67:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_96 Depth 2
                                        //       Child Loop BB10_99 Depth 3
                                        //     Child Loop BB10_73 Depth 2
                                        //       Child Loop BB10_78 Depth 3
                                        //     Child Loop BB10_81 Depth 2
                                        //       Child Loop BB10_83 Depth 3
                                        //     Child Loop BB10_88 Depth 2
	cmp	w1, #1
	b.le	.LBB10_84
// %bb.68:                              //   in Loop: Header=BB10_67 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x1, x10
	mov	x3, x12
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	cmp	w1, #17
	b.lt	.LBB10_91
// %bb.69:                              // %.preheader21
                                        //   in Loop: Header=BB10_67 Depth=1
	mov	x15, x11
	mov	w16, #8                         // =0x8
	mov	x17, x4
	b	.LBB10_73
.LBB10_70:                              //   in Loop: Header=BB10_73 Depth=2
	str	d1, [x18, #8]
.LBB10_71:                              //   in Loop: Header=BB10_73 Depth=2
	mov	x18, x4
.LBB10_72:                              //   in Loop: Header=BB10_73 Depth=2
	add	x16, x16, #8
	add	x15, x15, #8
	str	d0, [x18]
	cmp	x16, #128
	b.eq	.LBB10_79
.LBB10_73:                              //   Parent Loop BB10_67 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_78 Depth 3
	mov	x18, x17
	add	x17, x4, x16
	ldr	d1, [x4]
	ldr	d0, [x17]
	fcmp	d0, d1
	b.pl	.LBB10_76
// %bb.74:                              //   in Loop: Header=BB10_73 Depth=2
	cmp	x16, #9
	b.lo	.LBB10_70
// %bb.75:                              //   in Loop: Header=BB10_73 Depth=2
	mov	x0, x11
	mov	x1, x4
	mov	x2, x16
	bl	memmove
	b	.LBB10_71
.LBB10_76:                              //   in Loop: Header=BB10_73 Depth=2
	ldr	d1, [x18]
	mov	x18, x17
	fcmp	d0, d1
	b.pl	.LBB10_72
// %bb.77:                              // %.preheader14
                                        //   in Loop: Header=BB10_73 Depth=2
	mov	x18, x15
.LBB10_78:                              //   Parent Loop BB10_67 Depth=1
                                        //     Parent Loop BB10_73 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d1, [x18]
	ldur	d1, [x18, #-16]
	sub	x18, x18, #8
	fcmp	d0, d1
	b.mi	.LBB10_78
	b	.LBB10_72
.LBB10_79:                              // %.preheader20
                                        //   in Loop: Header=BB10_67 Depth=1
	mov	x15, x13
	b	.LBB10_81
.LBB10_80:                              //   in Loop: Header=BB10_81 Depth=2
	add	x15, x15, #8
	str	d0, [x16]
	cmp	x15, x10
	b.eq	.LBB10_87
.LBB10_81:                              //   Parent Loop BB10_67 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_83 Depth 3
	ldp	d1, d0, [x15, #-8]
	mov	x16, x15
	fcmp	d0, d1
	b.pl	.LBB10_80
// %bb.82:                              // %.preheader12
                                        //   in Loop: Header=BB10_81 Depth=2
	mov	x16, x15
.LBB10_83:                              //   Parent Loop BB10_67 Depth=1
                                        //     Parent Loop BB10_81 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d1, [x16]
	ldur	d1, [x16, #-16]
	sub	x16, x16, #8
	fcmp	d0, d1
	b.mi	.LBB10_83
	b	.LBB10_80
.LBB10_84:                              //   in Loop: Header=BB10_67 Depth=1
	cbz	w1, .LBB10_87
// %bb.85:                              //   in Loop: Header=BB10_67 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_90
// %bb.86:                              //   in Loop: Header=BB10_67 Depth=1
	ldr	d0, [x3]
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x1, x10
	mov	x3, x12
	str	d0, [x4]
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
.LBB10_87:                              // %.preheader18
                                        //   in Loop: Header=BB10_67 Depth=1
	mov	x15, x5
	mov	x16, x11
.LBB10_88:                              //   Parent Loop BB10_67 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x15, .LBB10_66
// %bb.89:                              //   in Loop: Header=BB10_88 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	sub	x15, x15, #8
	fcmp	d0, d1
	b.pl	.LBB10_88
	b	.LBB10_65
.LBB10_90:                              //   in Loop: Header=BB10_67 Depth=1
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x1, x10
	mov	x3, x12
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
.LBB10_91:                              // %.preheader22
                                        //   in Loop: Header=BB10_67 Depth=1
	mov	x18, x11
	mov	x15, x4
	b	.LBB10_96
.LBB10_92:                              //   in Loop: Header=BB10_96 Depth=2
	sub	x16, x15, x4
	asr	x18, x16, #3
	cmp	x18, #2
	b.lt	.LBB10_100
// %bb.93:                              //   in Loop: Header=BB10_96 Depth=2
	sub	x17, x17, x18, lsl #3
	mov	x1, x4
	mov	x2, x16
	add	x0, x17, #16
	bl	memmove
.LBB10_94:                              //   in Loop: Header=BB10_96 Depth=2
	mov	x16, x4
.LBB10_95:                              //   in Loop: Header=BB10_96 Depth=2
	add	x18, x15, #8
	str	d0, [x16]
	cmp	x18, x10
	b.eq	.LBB10_87
.LBB10_96:                              //   Parent Loop BB10_67 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_99 Depth 3
	ldr	d0, [x18]
	ldr	d1, [x4]
	mov	x17, x15
	mov	x15, x18
	fcmp	d0, d1
	b.mi	.LBB10_92
// %bb.97:                              //   in Loop: Header=BB10_96 Depth=2
	ldr	d1, [x17]
	mov	x16, x15
	fcmp	d0, d1
	b.pl	.LBB10_95
// %bb.98:                              // %.preheader16
                                        //   in Loop: Header=BB10_96 Depth=2
	mov	x16, x15
.LBB10_99:                              //   Parent Loop BB10_67 Depth=1
                                        //     Parent Loop BB10_96 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d1, [x16]
	ldur	d1, [x16, #-16]
	sub	x16, x16, #8
	fcmp	d0, d1
	b.mi	.LBB10_99
	b	.LBB10_95
.LBB10_100:                             //   in Loop: Header=BB10_96 Depth=2
	cmp	x16, #8
	mov	x16, x4
	b.ne	.LBB10_95
// %bb.101:                             //   in Loop: Header=BB10_96 Depth=2
	str	d1, [x17, #8]
	b	.LBB10_94
.LBB10_102:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_105
.LBB10_103:                             //   in Loop: Header=BB10_105 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_104:                             //   in Loop: Header=BB10_105 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_112
.LBB10_105:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_108 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_110
// %bb.106:                             //   in Loop: Header=BB10_105 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_107:                             //   in Loop: Header=BB10_105 Depth=1
	mov	x0, x4
	mov	x1, x7
	bl	_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	mov	x12, x5
	mov	x13, x10
.LBB10_108:                             //   Parent Loop BB10_105 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_104
// %bb.109:                             //   in Loop: Header=BB10_108 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_108
	b	.LBB10_103
.LBB10_110:                             //   in Loop: Header=BB10_105 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_107
// %bb.111:                             //   in Loop: Header=BB10_105 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_107
.LBB10_112:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
.LBB10_113:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_116 Depth 2
	cbz	w1, .LBB10_115
// %bb.114:                             //   in Loop: Header=BB10_113 Depth=1
	add	x1, x4, x2
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x3, xzr
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	add	x1, x4, x2
	mov	x0, x4
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_
.LBB10_115:                             // %.preheader9
                                        //   in Loop: Header=BB10_113 Depth=1
	mov	x12, x5
	mov	x13, x10
.LBB10_116:                             //   Parent Loop BB10_113 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_119
// %bb.117:                             //   in Loop: Header=BB10_116 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_116
// %bb.118:                             //   in Loop: Header=BB10_113 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_119:                             //   in Loop: Header=BB10_113 Depth=1
	cmp	w1, #2
	b.lt	.LBB10_122
// %bb.120:                             //   in Loop: Header=BB10_113 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_121:                             //   in Loop: Header=BB10_113 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.ne	.LBB10_113
	b	.LBB10_124
.LBB10_122:                             //   in Loop: Header=BB10_113 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_121
// %bb.123:                             //   in Loop: Header=BB10_113 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_121
.LBB10_124:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_127
.LBB10_125:                             //   in Loop: Header=BB10_127 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_126:                             //   in Loop: Header=BB10_127 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_134
.LBB10_127:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_130 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_132
// %bb.128:                             //   in Loop: Header=BB10_127 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_129:                             //   in Loop: Header=BB10_127 Depth=1
	mov	x0, x4
	mov	x1, x7
	bl	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	mov	x12, x5
	mov	x13, x10
.LBB10_130:                             //   Parent Loop BB10_127 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_126
// %bb.131:                             //   in Loop: Header=BB10_130 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_130
	b	.LBB10_125
.LBB10_132:                             //   in Loop: Header=BB10_127 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_129
// %bb.133:                             //   in Loop: Header=BB10_127 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_129
.LBB10_134:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_137
.LBB10_135:                             //   in Loop: Header=BB10_137 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_136:                             //   in Loop: Header=BB10_137 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_146
.LBB10_137:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_141 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_143
// %bb.138:                             //   in Loop: Header=BB10_137 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_139:                             //   in Loop: Header=BB10_137 Depth=1
	add	x1, x4, x2
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x3, xzr
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	add	x1, x4, x2
	mov	x0, x4
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_
.LBB10_140:                             // %.preheader6
                                        //   in Loop: Header=BB10_137 Depth=1
	mov	x12, x5
	mov	x13, x10
.LBB10_141:                             //   Parent Loop BB10_137 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_136
// %bb.142:                             //   in Loop: Header=BB10_141 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_141
	b	.LBB10_135
.LBB10_143:                             //   in Loop: Header=BB10_137 Depth=1
	cbz	w1, .LBB10_140
// %bb.144:                             //   in Loop: Header=BB10_137 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_139
// %bb.145:                             //   in Loop: Header=BB10_137 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_139
.LBB10_146:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_149
.LBB10_147:                             //   in Loop: Header=BB10_149 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_148:                             //   in Loop: Header=BB10_149 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_156
.LBB10_149:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_152 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_154
// %bb.150:                             //   in Loop: Header=BB10_149 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_151:                             //   in Loop: Header=BB10_149 Depth=1
	mov	x0, x4
	mov	x1, x7
	bl	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	mov	x12, x5
	mov	x13, x10
.LBB10_152:                             //   Parent Loop BB10_149 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_148
// %bb.153:                             //   in Loop: Header=BB10_152 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_152
	b	.LBB10_147
.LBB10_154:                             //   in Loop: Header=BB10_149 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_151
// %bb.155:                             //   in Loop: Header=BB10_149 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_151
.LBB10_156:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_159
.LBB10_157:                             //   in Loop: Header=BB10_159 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_158:                             //   in Loop: Header=BB10_159 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_168
.LBB10_159:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_163 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_165
// %bb.160:                             //   in Loop: Header=BB10_159 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_161:                             //   in Loop: Header=BB10_159 Depth=1
	add	x1, x4, x2
	eor	x2, x8, #0x7e
	mov	x0, x4
	mov	x3, xzr
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	add	x1, x4, x2
	mov	x0, x4
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_
.LBB10_162:                             // %.preheader3
                                        //   in Loop: Header=BB10_159 Depth=1
	mov	x12, x5
	mov	x13, x10
.LBB10_163:                             //   Parent Loop BB10_159 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_158
// %bb.164:                             //   in Loop: Header=BB10_163 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_163
	b	.LBB10_157
.LBB10_165:                             //   in Loop: Header=BB10_159 Depth=1
	cbz	w1, .LBB10_162
// %bb.166:                             //   in Loop: Header=BB10_159 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_161
// %bb.167:                             //   in Loop: Header=BB10_159 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_161
.LBB10_168:
	mov	w9, wzr
	add	x10, x4, #8
	adrp	x11, .L.str.11
	add	x11, x11, :lo12:.L.str.11
	b	.LBB10_171
.LBB10_169:                             //   in Loop: Header=BB10_171 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x11
	bl	printf
.LBB10_170:                             //   in Loop: Header=BB10_171 Depth=1
	add	w9, w9, #1
	cmp	w9, w0
	b.eq	.LBB10_178
.LBB10_171:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_174 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_176
// %bb.172:                             //   in Loop: Header=BB10_171 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_173:                             //   in Loop: Header=BB10_171 Depth=1
	mov	x0, x4
	mov	x1, x7
	bl	_Z9quicksortIPdEvT_S1_
	mov	x12, x5
	mov	x13, x10
.LBB10_174:                             //   Parent Loop BB10_171 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x12, .LBB10_170
// %bb.175:                             //   in Loop: Header=BB10_174 Depth=2
	ldp	d1, d0, [x13, #-8]
	add	x13, x13, #8
	sub	x12, x12, #8
	fcmp	d0, d1
	b.pl	.LBB10_174
	b	.LBB10_169
.LBB10_176:                             //   in Loop: Header=BB10_171 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_173
// %bb.177:                             //   in Loop: Header=BB10_171 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_173
.LBB10_178:
	mov	w7, wzr
	add	x9, x4, #8
	adrp	x10, .L.str.11
	add	x10, x10, :lo12:.L.str.11
	b	.LBB10_181
.LBB10_179:                             //   in Loop: Header=BB10_181 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x10
	bl	printf
.LBB10_180:                             //   in Loop: Header=BB10_181 Depth=1
	add	w7, w7, #1
	cmp	w7, w0
	b.eq	.LBB10_190
.LBB10_181:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_185 Depth 2
	cmp	w1, #2
	b.lt	.LBB10_187
// %bb.182:                             //   in Loop: Header=BB10_181 Depth=1
	mov	x0, x4
	mov	x1, x3
	bl	memcpy
.LBB10_183:                             //   in Loop: Header=BB10_181 Depth=1
	add	x1, x4, x2
	eor	x2, x8, #0x7e
	mov	x0, x4
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	add	x1, x4, x2
	mov	x0, x4
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_
.LBB10_184:                             // %.preheader
                                        //   in Loop: Header=BB10_181 Depth=1
	mov	x11, x5
	mov	x12, x9
.LBB10_185:                             //   Parent Loop BB10_181 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x11, .LBB10_180
// %bb.186:                             //   in Loop: Header=BB10_185 Depth=2
	ldp	d1, d0, [x12, #-8]
	add	x12, x12, #8
	sub	x11, x11, #8
	fcmp	d0, d1
	b.pl	.LBB10_185
	b	.LBB10_179
.LBB10_187:                             //   in Loop: Header=BB10_181 Depth=1
	cbz	w1, .LBB10_184
// %bb.188:                             //   in Loop: Header=BB10_181 Depth=1
	cmp	w1, #1
	b.ne	.LBB10_183
// %bb.189:                             //   in Loop: Header=BB10_181 Depth=1
	ldr	d0, [x3]
	str	d0, [x4]
	b	.LBB10_183
.LBB10_190:
	mov	x0, x4
	bl	_ZdaPv
	mov	x0, x3
	bl	_ZdaPv
	mov	w0, wzr
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end10:
	.size	main, .Lfunc_end10-main
	.cfi_endproc
                                        // -- End function
	.section	.text._Z9quicksortIPd17less_than_functorEvT_S2_T0_,"axG",@progbits,_Z9quicksortIPd17less_than_functorEvT_S2_T0_,comdat
	.weak	_Z9quicksortIPd17less_than_functorEvT_S2_T0_ // -- Begin function _Z9quicksortIPd17less_than_functorEvT_S2_T0_
	.p2align	2
	.type	_Z9quicksortIPd17less_than_functorEvT_S2_T0_,@function
_Z9quicksortIPd17less_than_functorEvT_S2_T0_: // @_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB11_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB11_3
.LBB11_2:                               //   in Loop: Header=BB11_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB11_11
.LBB11_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_4 Depth 2
                                        //       Child Loop BB11_5 Depth 3
                                        //       Child Loop BB11_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB11_4:                               //   Parent Loop BB11_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB11_5 Depth 3
                                        //       Child Loop BB11_8 Depth 3
	sub	x5, x2, x3
.LBB11_5:                               //   Parent Loop BB11_3 Depth=1
                                        //     Parent Loop BB11_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB11_5
// %bb.6:                               //   in Loop: Header=BB11_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB11_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB11_4 Depth=2
	sub	x4, x4, #8
.LBB11_8:                               //   Parent Loop BB11_3 Depth=1
                                        //     Parent Loop BB11_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB11_8
// %bb.9:                               //   in Loop: Header=BB11_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB11_2
// %bb.10:                              //   in Loop: Header=BB11_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB11_4
.LBB11_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB11_12:
	ret
.Lfunc_end11:
	.size	_Z9quicksortIPd17less_than_functorEvT_S2_T0_, .Lfunc_end11-_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_,"axG",@progbits,_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_,comdat
	.weak	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_ // -- Begin function _Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	.p2align	2
	.type	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_,@function
_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_: // @_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB12_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB12_3
.LBB12_2:                               //   in Loop: Header=BB12_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB12_11
.LBB12_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB12_4 Depth 2
                                        //       Child Loop BB12_5 Depth 3
                                        //       Child Loop BB12_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB12_4:                               //   Parent Loop BB12_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB12_5 Depth 3
                                        //       Child Loop BB12_8 Depth 3
	sub	x5, x2, x3
.LBB12_5:                               //   Parent Loop BB12_3 Depth=1
                                        //     Parent Loop BB12_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB12_5
// %bb.6:                               //   in Loop: Header=BB12_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB12_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB12_4 Depth=2
	sub	x4, x4, #8
.LBB12_8:                               //   Parent Loop BB12_3 Depth=1
                                        //     Parent Loop BB12_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB12_8
// %bb.9:                               //   in Loop: Header=BB12_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB12_2
// %bb.10:                              //   in Loop: Header=BB12_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB12_4
.LBB12_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB12_12:
	ret
.Lfunc_end12:
	.size	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_, .Lfunc_end12-_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._Z9quicksortIPdSt4lessIdEEvT_S3_T0_,"axG",@progbits,_Z9quicksortIPdSt4lessIdEEvT_S3_T0_,comdat
	.weak	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_ // -- Begin function _Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	.p2align	2
	.type	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_,@function
_Z9quicksortIPdSt4lessIdEEvT_S3_T0_:    // @_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB13_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB13_3
.LBB13_2:                               //   in Loop: Header=BB13_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB13_11
.LBB13_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_4 Depth 2
                                        //       Child Loop BB13_5 Depth 3
                                        //       Child Loop BB13_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB13_4:                               //   Parent Loop BB13_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB13_5 Depth 3
                                        //       Child Loop BB13_8 Depth 3
	sub	x5, x2, x3
.LBB13_5:                               //   Parent Loop BB13_3 Depth=1
                                        //     Parent Loop BB13_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB13_5
// %bb.6:                               //   in Loop: Header=BB13_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB13_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB13_4 Depth=2
	sub	x4, x4, #8
.LBB13_8:                               //   Parent Loop BB13_3 Depth=1
                                        //     Parent Loop BB13_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB13_8
// %bb.9:                               //   in Loop: Header=BB13_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB13_2
// %bb.10:                              //   in Loop: Header=BB13_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB13_4
.LBB13_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB13_12:
	ret
.Lfunc_end13:
	.size	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_, .Lfunc_end13-_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._Z9quicksortIPdEvT_S1_,"axG",@progbits,_Z9quicksortIPdEvT_S1_,comdat
	.weak	_Z9quicksortIPdEvT_S1_          // -- Begin function _Z9quicksortIPdEvT_S1_
	.p2align	2
	.type	_Z9quicksortIPdEvT_S1_,@function
_Z9quicksortIPdEvT_S1_:                 // @_Z9quicksortIPdEvT_S1_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB14_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB14_3
.LBB14_2:                               //   in Loop: Header=BB14_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_Z9quicksortIPdEvT_S1_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB14_11
.LBB14_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_4 Depth 2
                                        //       Child Loop BB14_5 Depth 3
                                        //       Child Loop BB14_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB14_4:                               //   Parent Loop BB14_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB14_5 Depth 3
                                        //       Child Loop BB14_8 Depth 3
	sub	x5, x2, x3
.LBB14_5:                               //   Parent Loop BB14_3 Depth=1
                                        //     Parent Loop BB14_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB14_5
// %bb.6:                               //   in Loop: Header=BB14_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB14_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB14_4 Depth=2
	sub	x4, x4, #8
.LBB14_8:                               //   Parent Loop BB14_3 Depth=1
                                        //     Parent Loop BB14_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB14_8
// %bb.9:                               //   in Loop: Header=BB14_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB14_2
// %bb.10:                              //   in Loop: Header=BB14_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB14_4
.LBB14_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB14_12:
	ret
.Lfunc_end14:
	.size	_Z9quicksortIPdEvT_S1_, .Lfunc_end14-_Z9quicksortIPdEvT_S1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_ // -- Begin function _ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	.p2align	2
	.type	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_,@function
_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_: // @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	.cfi_startproc
// %bb.0:
	mov	x0, x3
	mov	x2, x1
	sub	x6, x1, x0
	cmp	x6, #129
	b.lt	.LBB15_36
// %bb.1:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x5, #-8                         // =0xfffffffffffffff8
	mov	x1, x0
	mov	x3, x2
	add	x4, x0, #8
	sub	x5, x5, x0
	b	.LBB15_3
.LBB15_2:                               //   in Loop: Header=BB15_3 Depth=1
	mov	x0, x8
	mov	x1, x2
	mov	x2, x3
	mov	x3, x8
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	cmp	x6, #128
	mov	x2, x8
	b.le	.LBB15_35
.LBB15_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_16 Depth 2
                                        //       Child Loop BB15_17 Depth 3
                                        //       Child Loop BB15_19 Depth 3
	cbz	x3, .LBB15_22
// %bb.4:                               //   in Loop: Header=BB15_3 Depth=1
	lsr	x6, x6, #4
	ldr	d1, [x1, #8]
	ldr	d0, [x1, x6, lsl #3]
	fmov	d0, d1
	blr	x0
	ldur	d0, [x2, #-8]
	tbz	w0, #0, .LBB15_7
// %bb.5:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x1, x6, lsl #3]
	fmov	d1, d0
	blr	x0
	tbz	w0, #0, .LBB15_9
// %bb.6:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x1, x6, lsl #3]
	ldr	d0, [x1]
	b	.LBB15_14
.LBB15_7:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x4]
	fmov	d1, d0
	blr	x0
	tbz	w0, #0, .LBB15_11
// %bb.8:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	q0, [x1]
	ext	v0.16b, v0.16b, v0.16b, #8
	str	q0, [x1]
	b	.LBB15_15
.LBB15_9:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x1, #8]
	ldur	d0, [x2, #-8]
	fmov	d0, d1
	blr	x0
	ldr	d0, [x1]
	tbnz	w0, #0, .LBB15_12
// %bb.10:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x1, #8]
	stp	d1, d0, [x1]
	b	.LBB15_15
.LBB15_11:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x1, x6, lsl #3]
	ldur	d1, [x2, #-8]
	blr	x0
	ldr	d0, [x1]
	tbz	w0, #0, .LBB15_13
.LBB15_12:                              //   in Loop: Header=BB15_3 Depth=1
	ldur	d1, [x2, #-8]
	str	d1, [x1]
	stur	d0, [x2, #-8]
	b	.LBB15_15
.LBB15_13:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x1, x6, lsl #3]
.LBB15_14:                              // %.preheader6
                                        //   in Loop: Header=BB15_3 Depth=1
	str	d1, [x1]
	str	d0, [x1, x6, lsl #3]
.LBB15_15:                              // %.preheader6
                                        //   in Loop: Header=BB15_3 Depth=1
	sub	x3, x3, #1
	mov	x7, x2
	mov	x9, x4
.LBB15_16:                              //   Parent Loop BB15_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB15_17 Depth 3
                                        //       Child Loop BB15_19 Depth 3
	add	x6, x5, x9
	sub	x8, x9, #8
.LBB15_17:                              //   Parent Loop BB15_3 Depth=1
                                        //     Parent Loop BB15_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8, #8]!
	ldr	d0, [x1]
	fmov	d0, d1
	blr	x0
	add	x6, x6, #8
	tbnz	w0, #0, .LBB15_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB15_16 Depth=2
	add	x9, x8, #8
.LBB15_19:                              //   Parent Loop BB15_3 Depth=1
                                        //     Parent Loop BB15_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d0, [x7, #-8]!
	ldr	d0, [x1]
	fmov	d1, d0
	blr	x0
	tbnz	w0, #0, .LBB15_19
// %bb.20:                              //   in Loop: Header=BB15_16 Depth=2
	cmp	x8, x7
	b.hs	.LBB15_2
// %bb.21:                              //   in Loop: Header=BB15_16 Depth=2
	ldr	d1, [x7]
	ldr	d0, [x8]
	str	d1, [x8]
	str	d0, [x7]
	b	.LBB15_16
.LBB15_22:
	add	x2, sp, #8
	str	x0, [sp, #8]
	mov	x0, x1
	mov	x1, x2
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_
	mov	w3, #1                          // =0x1
	b	.LBB15_25
.LBB15_23:                              //   in Loop: Header=BB15_25 Depth=1
	mov	x5, xzr
.LBB15_24:                              //   in Loop: Header=BB15_25 Depth=1
	cmp	x4, #8
	str	d0, [x1, x5, lsl #3]
	b.le	.LBB15_35
.LBB15_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_27 Depth 2
                                        //     Child Loop BB15_33 Depth 2
	ldr	d0, [x2, #-8]!
	sub	x4, x2, x1
	asr	x6, x4, #3
	ldr	d1, [x1]
	cmp	x6, #3
	str	d1, [x2]
	b.lt	.LBB15_29
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB15_25 Depth=1
	sub	x5, x6, #1
	mov	x8, xzr
	add	x5, x5, x5, lsr #63
	asr	x7, x5, #1
.LBB15_27:                              //   Parent Loop BB15_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x5, x8, #1
	add	x10, x1, x8, lsl #4
	add	x9, x5, #2
	ldr	d1, [x10, #8]
	ldr	d0, [x1, x9, lsl #3]
	blr	x0
	orr	x5, x5, #0x1
	tst	w0, #0x1
	csel	x5, x9, x5, eq
	ldr	d1, [x1, x5, lsl #3]
	cmp	x5, x7
	str	d1, [x1, x8, lsl #3]
	mov	x8, x5
	b.lt	.LBB15_27
// %bb.28:                              //   in Loop: Header=BB15_25 Depth=1
	tbz	w4, #3, .LBB15_30
	b	.LBB15_32
.LBB15_29:                              //   in Loop: Header=BB15_25 Depth=1
	mov	x5, xzr
	tbnz	w4, #3, .LBB15_32
.LBB15_30:                              //   in Loop: Header=BB15_25 Depth=1
	sub	x6, x6, #2
	cmp	x5, x6, asr #1
	b.ne	.LBB15_32
// %bb.31:                              //   in Loop: Header=BB15_25 Depth=1
	orr	x6, x3, x5, lsl #1
	ldr	d1, [x1, x6, lsl #3]
	str	d1, [x1, x5, lsl #3]
	mov	x5, x6
	b	.LBB15_33
.LBB15_32:                              //   in Loop: Header=BB15_25 Depth=1
	cbz	x5, .LBB15_24
.LBB15_33:                              //   Parent Loop BB15_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x7, x5, #1
	lsr	x6, x7, #1
	ldr	d0, [x1, x6, lsl #3]
	fmov	d1, d0
	blr	x0
	tbz	w0, #0, .LBB15_24
// %bb.34:                              //   in Loop: Header=BB15_33 Depth=2
	ldr	d1, [x1, x6, lsl #3]
	cmp	x7, #1
	str	d1, [x1, x5, lsl #3]
	mov	x5, x6
	b.hi	.LBB15_33
	b	.LBB15_23
.LBB15_35:
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB15_36:
	ret
.Lfunc_end15:
	.size	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_, .Lfunc_end15-_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_,"axG",@progbits,_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_,comdat
	.weak	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_ // -- Begin function _ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_
	.p2align	2
	.type	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_,@function
_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_: // @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_
	.cfi_startproc
// %bb.0:
	sub	x5, x1, x0
	asr	x2, x5, #3
	subs	x4, x2, #2
	b.lt	.LBB16_22
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x1, x2
	sub	x2, x2, #1
	lsr	x3, x4, #1
	lsr	x2, x2, #1
	tbnz	w5, #3, .LBB16_15
// %bb.2:
	orr	x4, x4, #0x1
	mov	x6, x3
	b	.LBB16_5
.LBB16_3:                               //   in Loop: Header=BB16_5 Depth=1
	mov	x8, x7
.LBB16_4:                               //   in Loop: Header=BB16_5 Depth=1
	sub	x6, x5, #1
	str	d0, [x0, x8, lsl #3]
	cbz	x5, .LBB16_21
.LBB16_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_7 Depth 2
                                        //     Child Loop BB16_11 Depth 2
	mov	x5, x6
	ldr	d0, [x0, x6, lsl #3]
	ldr	x6, [x1]
	cmp	x5, x2
	mov	x7, x5
	b.ge	.LBB16_8
// %bb.6:                               // %.preheader1
                                        //   in Loop: Header=BB16_5 Depth=1
	mov	x8, x5
.LBB16_7:                               //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x8, #1
	add	x10, x0, x8, lsl #4
	add	x9, x7, #2
	ldr	d1, [x10, #8]
	ldr	d0, [x0, x9, lsl #3]
	blr	x6
	orr	x7, x7, #0x1
	tst	w0, #0x1
	csel	x7, x9, x7, eq
	ldr	d1, [x0, x7, lsl #3]
	cmp	x7, x2
	str	d1, [x0, x8, lsl #3]
	mov	x8, x7
	b.lt	.LBB16_7
.LBB16_8:                               //   in Loop: Header=BB16_5 Depth=1
	cmp	x7, x3
	b.ne	.LBB16_10
// %bb.9:                               //   in Loop: Header=BB16_5 Depth=1
	ldr	d1, [x0, x4, lsl #3]
	mov	x7, x4
	str	d1, [x0, x3, lsl #3]
.LBB16_10:                              //   in Loop: Header=BB16_5 Depth=1
	cmp	x7, x5
	b.le	.LBB16_3
.LBB16_11:                              //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x8, x7, #1
	add	x8, x8, x8, lsr #63
	asr	x8, x8, #1
	ldr	d0, [x0, x8, lsl #3]
	fmov	d1, d0
	blr	x6
	tbz	w0, #0, .LBB16_3
// %bb.12:                              //   in Loop: Header=BB16_11 Depth=2
	ldr	d1, [x0, x8, lsl #3]
	cmp	x8, x5
	str	d1, [x0, x7, lsl #3]
	mov	x7, x8
	b.gt	.LBB16_11
	b	.LBB16_4
.LBB16_13:                              //   in Loop: Header=BB16_15 Depth=1
	mov	x6, x5
.LBB16_14:                              //   in Loop: Header=BB16_15 Depth=1
	sub	x3, x4, #1
	str	d0, [x0, x6, lsl #3]
	cbz	x4, .LBB16_21
.LBB16_15:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_17 Depth 2
                                        //     Child Loop BB16_19 Depth 2
	ldr	d0, [x0, x3, lsl #3]
	mov	x4, x3
	cmp	x3, x2
	mov	x6, x3
	b.ge	.LBB16_14
// %bb.16:                              // %.preheader7
                                        //   in Loop: Header=BB16_15 Depth=1
	ldr	x3, [x1]
	mov	x5, x4
.LBB16_17:                              //   Parent Loop BB16_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	mov	x6, x5
	lsl	x5, x5, #1
	add	x8, x0, x6, lsl #4
	add	x7, x5, #2
	ldr	d0, [x0, x7, lsl #3]
	ldr	d1, [x8, #8]
	blr	x3
	orr	x5, x5, #0x1
	tst	w0, #0x1
	csel	x5, x7, x5, eq
	ldr	d1, [x0, x5, lsl #3]
	cmp	x5, x2
	str	d1, [x0, x6, lsl #3]
	b.lt	.LBB16_17
// %bb.18:                              //   in Loop: Header=BB16_15 Depth=1
	cmp	x5, x4
	b.le	.LBB16_13
.LBB16_19:                              //   Parent Loop BB16_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d0, [x0, x6, lsl #3]
	fmov	d1, d0
	blr	x3
	tbz	w0, #0, .LBB16_13
// %bb.20:                              //   in Loop: Header=BB16_19 Depth=2
	ldr	d1, [x0, x6, lsl #3]
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.gt	.LBB16_19
	b	.LBB16_14
.LBB16_21:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB16_22:
	ret
.Lfunc_end16:
	.size	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_, .Lfunc_end16-_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_ // -- Begin function _ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	.p2align	2
	.type	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_,@function
_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_: // @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	.cfi_startproc
// %bb.0:
	sub	x6, x1, x0
	cmp	x6, #129
	b.lt	.LBB17_38
// %bb.1:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x5, #-8                         // =0xfffffffffffffff8
	mov	x2, x3
	add	x4, x0, #8
	sub	x5, x5, x0
	b	.LBB17_3
.LBB17_2:                               //   in Loop: Header=BB17_3 Depth=1
	and	x3, x2, #0xff
	mov	x0, x9
	mov	x2, x3
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	cmp	x6, #128
	mov	x1, x9
	b.le	.LBB17_37
.LBB17_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_16 Depth 2
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_19 Depth 3
	cbz	x3, .LBB17_22
// %bb.4:                               //   in Loop: Header=BB17_3 Depth=1
	lsr	x6, x6, #4
	ldr	d1, [x0, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x0, x6, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB17_7
// %bb.5:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB17_9
// %bb.6:                               //   in Loop: Header=BB17_3 Depth=1
	ldr	d0, [x0]
	str	d2, [x0]
	str	d0, [x0, x6, lsl #3]
	b	.LBB17_15
.LBB17_7:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB17_11
// %bb.8:                               //   in Loop: Header=BB17_3 Depth=1
	ldr	d0, [x0]
	stp	d1, d0, [x0]
	b	.LBB17_15
.LBB17_9:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x0]
	b.pl	.LBB17_13
// %bb.10:                              //   in Loop: Header=BB17_3 Depth=1
	str	d0, [x0]
	stur	d2, [x1, #-8]
	b	.LBB17_15
.LBB17_11:                              //   in Loop: Header=BB17_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x0]
	b.pl	.LBB17_14
// %bb.12:                              //   in Loop: Header=BB17_3 Depth=1
	str	d0, [x0]
	stur	d1, [x1, #-8]
	b	.LBB17_15
.LBB17_13:                              //   in Loop: Header=BB17_3 Depth=1
	stp	d1, d2, [x0]
	b	.LBB17_15
.LBB17_14:                              //   in Loop: Header=BB17_3 Depth=1
	str	d2, [x0]
	str	d1, [x0, x6, lsl #3]
.LBB17_15:                              // %.preheader6
                                        //   in Loop: Header=BB17_3 Depth=1
	sub	x3, x3, #1
	mov	x7, x1
	mov	x8, x4
.LBB17_16:                              //   Parent Loop BB17_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_19 Depth 3
	ldr	d0, [x0]
	add	x6, x5, x8
.LBB17_17:                              //   Parent Loop BB17_3 Depth=1
                                        //     Parent Loop BB17_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8], #8
	add	x6, x6, #8
	fcmp	d1, d0
	b.mi	.LBB17_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB17_16 Depth=2
	sub	x9, x8, #8
.LBB17_19:                              //   Parent Loop BB17_3 Depth=1
                                        //     Parent Loop BB17_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x7, #-8]!
	fcmp	d0, d2
	b.mi	.LBB17_19
// %bb.20:                              //   in Loop: Header=BB17_16 Depth=2
	cmp	x9, x7
	b.hs	.LBB17_2
// %bb.21:                              //   in Loop: Header=BB17_16 Depth=2
	str	d2, [x9]
	str	d1, [x7]
	b	.LBB17_16
.LBB17_22:
	sturb	w2, [x29, #-4]
	sub	x2, x29, #4
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_
	mov	w2, #1                          // =0x1
	b	.LBB17_25
.LBB17_23:                              //   in Loop: Header=BB17_25 Depth=1
	mov	x4, xzr
.LBB17_24:                              //   in Loop: Header=BB17_25 Depth=1
	cmp	x3, #8
	str	d0, [x0, x4, lsl #3]
	b.le	.LBB17_37
.LBB17_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_28 Depth 2
                                        //     Child Loop BB17_35 Depth 2
	ldr	d0, [x1, #-8]!
	sub	x3, x1, x0
	asr	x5, x3, #3
	ldr	d1, [x0]
	cmp	x5, #3
	str	d1, [x1]
	b.lt	.LBB17_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB17_25 Depth=1
	sub	x4, x5, #1
	mov	x7, xzr
	add	x4, x4, x4, lsr #63
	asr	x6, x4, #1
	b	.LBB17_28
.LBB17_27:                              // %select.end
                                        //   in Loop: Header=BB17_28 Depth=2
	ldr	d1, [x0, x4, lsl #3]
	cmp	x4, x6
	str	d1, [x0, x7, lsl #3]
	mov	x7, x4
	b.ge	.LBB17_31
.LBB17_28:                              //   Parent Loop BB17_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x8, x7, #1
	add	x9, x0, x7, lsl #4
	add	x4, x8, #2
	ldr	d2, [x9, #8]
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB17_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB17_28 Depth=2
	orr	x4, x8, #0x1
	b	.LBB17_27
.LBB17_30:                              //   in Loop: Header=BB17_25 Depth=1
	mov	x4, xzr
.LBB17_31:                              //   in Loop: Header=BB17_25 Depth=1
	tbnz	w3, #3, .LBB17_34
// %bb.32:                              //   in Loop: Header=BB17_25 Depth=1
	sub	x5, x5, #2
	cmp	x4, x5, asr #1
	b.ne	.LBB17_34
// %bb.33:                              //   in Loop: Header=BB17_25 Depth=1
	orr	x5, x2, x4, lsl #1
	ldr	d1, [x0, x5, lsl #3]
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b	.LBB17_35
.LBB17_34:                              //   in Loop: Header=BB17_25 Depth=1
	cbz	x4, .LBB17_24
.LBB17_35:                              //   Parent Loop BB17_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x4, #1
	lsr	x5, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB17_24
// %bb.36:                              //   in Loop: Header=BB17_35 Depth=2
	cmp	x6, #1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b.hi	.LBB17_35
	b	.LBB17_23
.LBB17_37:
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB17_38:
	ret
.Lfunc_end17:
	.size	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_, .Lfunc_end17-_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_ // -- Begin function _ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_
	.p2align	2
	.type	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_,@function
_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_: // @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	sub	x2, x1, x1
	mov	x0, x1
	cmp	x2, #129
	b.lt	.LBB18_2
// %bb.1:
	add	x2, x1, #8
	mov	w3, #8                          // =0x8
	mov	x5, x1
	mov	x4, x2
	b	.LBB18_18
.LBB18_2:
	cmp	x1, x0
	b.eq	.LBB18_25
// %bb.3:
	add	x5, x1, #8
	cmp	x5, x0
	b.eq	.LBB18_25
// %bb.4:                               // %.preheader7
	mov	x2, x1
	b	.LBB18_9
.LBB18_5:                               //   in Loop: Header=BB18_9 Depth=1
	sub	x3, x2, x1
	asr	x5, x3, #3
	cmp	x5, #2
	b.lt	.LBB18_13
// %bb.6:                               //   in Loop: Header=BB18_9 Depth=1
	sub	x4, x4, x5, lsl #3
	mov	x2, x3
	add	x0, x4, #16
	bl	memmove
.LBB18_7:                               //   in Loop: Header=BB18_9 Depth=1
	mov	x3, x1
.LBB18_8:                               //   in Loop: Header=BB18_9 Depth=1
	add	x5, x2, #8
	str	d0, [x3]
	cmp	x5, x0
	b.eq	.LBB18_25
.LBB18_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_12 Depth 2
	ldr	d0, [x2, #8]
	ldr	d1, [x1]
	mov	x4, x2
	mov	x2, x5
	fcmp	d0, d1
	b.mi	.LBB18_5
// %bb.10:                              //   in Loop: Header=BB18_9 Depth=1
	ldr	d1, [x4]
	mov	x3, x2
	fcmp	d0, d1
	b.pl	.LBB18_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB18_9 Depth=1
	mov	x3, x2
.LBB18_12:                              //   Parent Loop BB18_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x3]
	ldur	d1, [x3, #-16]
	sub	x3, x3, #8
	fcmp	d0, d1
	b.mi	.LBB18_12
	b	.LBB18_8
.LBB18_13:                              //   in Loop: Header=BB18_9 Depth=1
	cmp	x3, #8
	mov	x3, x1
	b.ne	.LBB18_8
// %bb.14:                              //   in Loop: Header=BB18_9 Depth=1
	str	d1, [x4, #8]
	b	.LBB18_7
.LBB18_15:                              //   in Loop: Header=BB18_18 Depth=1
	str	d1, [x6, #8]
.LBB18_16:                              //   in Loop: Header=BB18_18 Depth=1
	mov	x6, x1
.LBB18_17:                              //   in Loop: Header=BB18_18 Depth=1
	add	x3, x3, #8
	add	x4, x4, #8
	str	d0, [x6]
	cmp	x3, #128
	b.eq	.LBB18_24
.LBB18_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_23 Depth 2
	mov	x6, x5
	add	x5, x1, x3
	ldr	d1, [x1]
	ldr	d0, [x5]
	fcmp	d0, d1
	b.pl	.LBB18_21
// %bb.19:                              //   in Loop: Header=BB18_18 Depth=1
	cmp	x3, #9
	b.lo	.LBB18_15
// %bb.20:                              //   in Loop: Header=BB18_18 Depth=1
	mov	x0, x2
	mov	x2, x3
	bl	memmove
	b	.LBB18_16
.LBB18_21:                              //   in Loop: Header=BB18_18 Depth=1
	ldr	d1, [x6]
	mov	x6, x5
	fcmp	d0, d1
	b.pl	.LBB18_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB18_18 Depth=1
	mov	x6, x4
.LBB18_23:                              //   Parent Loop BB18_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x6]
	ldur	d1, [x6, #-16]
	sub	x6, x6, #8
	fcmp	d0, d1
	b.mi	.LBB18_23
	b	.LBB18_17
.LBB18_24:
	add	x1, x1, #128
	cmp	x1, x0
	b.ne	.LBB18_27
.LBB18_25:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB18_26:                              //   in Loop: Header=BB18_27 Depth=1
	.cfi_restore_state
	add	x1, x1, #8
	str	d0, [x2]
	cmp	x1, x0
	b.eq	.LBB18_25
.LBB18_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_29 Depth 2
	ldp	d1, d0, [x1, #-8]
	mov	x2, x1
	fcmp	d0, d1
	b.pl	.LBB18_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB18_27 Depth=1
	mov	x2, x1
.LBB18_29:                              //   Parent Loop BB18_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x2]
	ldur	d1, [x2, #-16]
	sub	x2, x2, #8
	fcmp	d0, d1
	b.mi	.LBB18_29
	b	.LBB18_26
.Lfunc_end18:
	.size	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_, .Lfunc_end18-_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_,"axG",@progbits,_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_,comdat
	.weak	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_ // -- Begin function _ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_
	.p2align	2
	.type	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_,@function
_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_: // @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_
	.cfi_startproc
// %bb.0:
	sub	x4, x1, x0
	asr	x1, x4, #3
	subs	x3, x1, #2
	b.ge	.LBB19_2
.LBB19_1:
	ret
.LBB19_2:
	sub	x1, x1, #1
	lsr	x2, x3, #1
	lsr	x1, x1, #1
	tbnz	w4, #3, .LBB19_18
// %bb.3:
	orr	x3, x3, #0x1
	mov	x5, x2
	b	.LBB19_6
.LBB19_4:                               //   in Loop: Header=BB19_6 Depth=1
	mov	x6, x5
.LBB19_5:                               //   in Loop: Header=BB19_6 Depth=1
	sub	x5, x4, #1
	str	d0, [x0, x6, lsl #3]
	cbz	x4, .LBB19_1
.LBB19_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_9 Depth 2
                                        //     Child Loop BB19_14 Depth 2
	ldr	d0, [x0, x5, lsl #3]
	mov	x4, x5
	cmp	x5, x1
	b.ge	.LBB19_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB19_6 Depth=1
	mov	x6, x4
	b	.LBB19_9
.LBB19_8:                               // %select.end
                                        //   in Loop: Header=BB19_9 Depth=2
	ldr	d1, [x0, x5, lsl #3]
	cmp	x5, x1
	str	d1, [x0, x6, lsl #3]
	mov	x6, x5
	b.ge	.LBB19_11
.LBB19_9:                               //   Parent Loop BB19_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x6, #1
	add	x8, x0, x6, lsl #4
	add	x5, x7, #2
	ldr	d2, [x8, #8]
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB19_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB19_9 Depth=2
	orr	x5, x7, #0x1
	b	.LBB19_8
.LBB19_11:                              //   in Loop: Header=BB19_6 Depth=1
	cmp	x5, x2
	b.ne	.LBB19_13
// %bb.12:                              //   in Loop: Header=BB19_6 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	mov	x5, x3
	str	d1, [x0, x2, lsl #3]
.LBB19_13:                              //   in Loop: Header=BB19_6 Depth=1
	cmp	x5, x4
	b.le	.LBB19_4
.LBB19_14:                              //   Parent Loop BB19_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_4
// %bb.15:                              //   in Loop: Header=BB19_14 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.gt	.LBB19_14
	b	.LBB19_5
.LBB19_16:                              //   in Loop: Header=BB19_18 Depth=1
	mov	x4, x2
.LBB19_17:                              //   in Loop: Header=BB19_18 Depth=1
	sub	x2, x3, #1
	str	d0, [x0, x4, lsl #3]
	cbz	x3, .LBB19_1
.LBB19_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_21 Depth 2
                                        //     Child Loop BB19_24 Depth 2
	ldr	d0, [x0, x2, lsl #3]
	mov	x3, x2
	cmp	x2, x1
	mov	x4, x2
	b.ge	.LBB19_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB19_18 Depth=1
	mov	x4, x3
	b	.LBB19_21
.LBB19_20:                              // %select.end10
                                        //   in Loop: Header=BB19_21 Depth=2
	ldr	d1, [x0, x2, lsl #3]
	cmp	x2, x1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x2
	b.ge	.LBB19_23
.LBB19_21:                              //   Parent Loop BB19_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x5, x4, #1
	add	x6, x0, x4, lsl #4
	add	x2, x5, #2
	ldr	d2, [x6, #8]
	ldr	d1, [x0, x2, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB19_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB19_21 Depth=2
	orr	x2, x5, #0x1
	b	.LBB19_20
.LBB19_23:                              //   in Loop: Header=BB19_18 Depth=1
	cmp	x2, x3
	b.le	.LBB19_16
.LBB19_24:                              //   Parent Loop BB19_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x4, x2, #1
	add	x4, x4, x4, lsr #63
	asr	x4, x4, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_16
// %bb.25:                              //   in Loop: Header=BB19_24 Depth=2
	cmp	x4, x3
	str	d1, [x0, x2, lsl #3]
	mov	x2, x4
	b.gt	.LBB19_24
	b	.LBB19_17
.Lfunc_end19:
	.size	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_, .Lfunc_end19-_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_ // -- Begin function _ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	.p2align	2
	.type	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_,@function
_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_: // @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	.cfi_startproc
// %bb.0:
	sub	x6, x1, x0
	cmp	x6, #129
	b.lt	.LBB20_38
// %bb.1:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x5, #-8                         // =0xfffffffffffffff8
	mov	x2, x3
	add	x4, x0, #8
	sub	x5, x5, x0
	b	.LBB20_3
.LBB20_2:                               //   in Loop: Header=BB20_3 Depth=1
	and	x3, x2, #0xff
	mov	x0, x9
	mov	x2, x3
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	cmp	x6, #128
	mov	x1, x9
	b.le	.LBB20_37
.LBB20_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_16 Depth 2
                                        //       Child Loop BB20_17 Depth 3
                                        //       Child Loop BB20_19 Depth 3
	cbz	x3, .LBB20_22
// %bb.4:                               //   in Loop: Header=BB20_3 Depth=1
	lsr	x6, x6, #4
	ldr	d1, [x0, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x0, x6, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB20_7
// %bb.5:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB20_9
// %bb.6:                               //   in Loop: Header=BB20_3 Depth=1
	ldr	d0, [x0]
	str	d2, [x0]
	str	d0, [x0, x6, lsl #3]
	b	.LBB20_15
.LBB20_7:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB20_11
// %bb.8:                               //   in Loop: Header=BB20_3 Depth=1
	ldr	d0, [x0]
	stp	d1, d0, [x0]
	b	.LBB20_15
.LBB20_9:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x0]
	b.pl	.LBB20_13
// %bb.10:                              //   in Loop: Header=BB20_3 Depth=1
	str	d0, [x0]
	stur	d2, [x1, #-8]
	b	.LBB20_15
.LBB20_11:                              //   in Loop: Header=BB20_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x0]
	b.pl	.LBB20_14
// %bb.12:                              //   in Loop: Header=BB20_3 Depth=1
	str	d0, [x0]
	stur	d1, [x1, #-8]
	b	.LBB20_15
.LBB20_13:                              //   in Loop: Header=BB20_3 Depth=1
	stp	d1, d2, [x0]
	b	.LBB20_15
.LBB20_14:                              //   in Loop: Header=BB20_3 Depth=1
	str	d2, [x0]
	str	d1, [x0, x6, lsl #3]
.LBB20_15:                              // %.preheader6
                                        //   in Loop: Header=BB20_3 Depth=1
	sub	x3, x3, #1
	mov	x7, x1
	mov	x8, x4
.LBB20_16:                              //   Parent Loop BB20_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB20_17 Depth 3
                                        //       Child Loop BB20_19 Depth 3
	ldr	d0, [x0]
	add	x6, x5, x8
.LBB20_17:                              //   Parent Loop BB20_3 Depth=1
                                        //     Parent Loop BB20_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8], #8
	add	x6, x6, #8
	fcmp	d1, d0
	b.mi	.LBB20_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB20_16 Depth=2
	sub	x9, x8, #8
.LBB20_19:                              //   Parent Loop BB20_3 Depth=1
                                        //     Parent Loop BB20_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x7, #-8]!
	fcmp	d0, d2
	b.mi	.LBB20_19
// %bb.20:                              //   in Loop: Header=BB20_16 Depth=2
	cmp	x9, x7
	b.hs	.LBB20_2
// %bb.21:                              //   in Loop: Header=BB20_16 Depth=2
	str	d2, [x9]
	str	d1, [x7]
	b	.LBB20_16
.LBB20_22:
	sturb	w2, [x29, #-4]
	sub	x2, x29, #4
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_
	mov	w2, #1                          // =0x1
	b	.LBB20_25
.LBB20_23:                              //   in Loop: Header=BB20_25 Depth=1
	mov	x4, xzr
.LBB20_24:                              //   in Loop: Header=BB20_25 Depth=1
	cmp	x3, #8
	str	d0, [x0, x4, lsl #3]
	b.le	.LBB20_37
.LBB20_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_28 Depth 2
                                        //     Child Loop BB20_35 Depth 2
	ldr	d0, [x1, #-8]!
	sub	x3, x1, x0
	asr	x5, x3, #3
	ldr	d1, [x0]
	cmp	x5, #3
	str	d1, [x1]
	b.lt	.LBB20_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB20_25 Depth=1
	sub	x4, x5, #1
	mov	x7, xzr
	add	x4, x4, x4, lsr #63
	asr	x6, x4, #1
	b	.LBB20_28
.LBB20_27:                              // %select.end
                                        //   in Loop: Header=BB20_28 Depth=2
	ldr	d1, [x0, x4, lsl #3]
	cmp	x4, x6
	str	d1, [x0, x7, lsl #3]
	mov	x7, x4
	b.ge	.LBB20_31
.LBB20_28:                              //   Parent Loop BB20_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x8, x7, #1
	add	x9, x0, x7, lsl #4
	add	x4, x8, #2
	ldr	d2, [x9, #8]
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB20_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB20_28 Depth=2
	orr	x4, x8, #0x1
	b	.LBB20_27
.LBB20_30:                              //   in Loop: Header=BB20_25 Depth=1
	mov	x4, xzr
.LBB20_31:                              //   in Loop: Header=BB20_25 Depth=1
	tbnz	w3, #3, .LBB20_34
// %bb.32:                              //   in Loop: Header=BB20_25 Depth=1
	sub	x5, x5, #2
	cmp	x4, x5, asr #1
	b.ne	.LBB20_34
// %bb.33:                              //   in Loop: Header=BB20_25 Depth=1
	orr	x5, x2, x4, lsl #1
	ldr	d1, [x0, x5, lsl #3]
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b	.LBB20_35
.LBB20_34:                              //   in Loop: Header=BB20_25 Depth=1
	cbz	x4, .LBB20_24
.LBB20_35:                              //   Parent Loop BB20_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x4, #1
	lsr	x5, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB20_24
// %bb.36:                              //   in Loop: Header=BB20_35 Depth=2
	cmp	x6, #1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b.hi	.LBB20_35
	b	.LBB20_23
.LBB20_37:
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB20_38:
	ret
.Lfunc_end20:
	.size	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_, .Lfunc_end20-_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_ // -- Begin function _ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_
	.p2align	2
	.type	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_,@function
_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_: // @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	sub	x2, x1, x1
	mov	x0, x1
	cmp	x2, #129
	b.lt	.LBB21_2
// %bb.1:
	add	x2, x1, #8
	mov	w3, #8                          // =0x8
	mov	x5, x1
	mov	x4, x2
	b	.LBB21_18
.LBB21_2:
	cmp	x1, x0
	b.eq	.LBB21_25
// %bb.3:
	add	x5, x1, #8
	cmp	x5, x0
	b.eq	.LBB21_25
// %bb.4:                               // %.preheader7
	mov	x2, x1
	b	.LBB21_9
.LBB21_5:                               //   in Loop: Header=BB21_9 Depth=1
	sub	x3, x2, x1
	asr	x5, x3, #3
	cmp	x5, #2
	b.lt	.LBB21_13
// %bb.6:                               //   in Loop: Header=BB21_9 Depth=1
	sub	x4, x4, x5, lsl #3
	mov	x2, x3
	add	x0, x4, #16
	bl	memmove
.LBB21_7:                               //   in Loop: Header=BB21_9 Depth=1
	mov	x3, x1
.LBB21_8:                               //   in Loop: Header=BB21_9 Depth=1
	add	x5, x2, #8
	str	d0, [x3]
	cmp	x5, x0
	b.eq	.LBB21_25
.LBB21_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_12 Depth 2
	ldr	d0, [x2, #8]
	ldr	d1, [x1]
	mov	x4, x2
	mov	x2, x5
	fcmp	d0, d1
	b.mi	.LBB21_5
// %bb.10:                              //   in Loop: Header=BB21_9 Depth=1
	ldr	d1, [x4]
	mov	x3, x2
	fcmp	d0, d1
	b.pl	.LBB21_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB21_9 Depth=1
	mov	x3, x2
.LBB21_12:                              //   Parent Loop BB21_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x3]
	ldur	d1, [x3, #-16]
	sub	x3, x3, #8
	fcmp	d0, d1
	b.mi	.LBB21_12
	b	.LBB21_8
.LBB21_13:                              //   in Loop: Header=BB21_9 Depth=1
	cmp	x3, #8
	mov	x3, x1
	b.ne	.LBB21_8
// %bb.14:                              //   in Loop: Header=BB21_9 Depth=1
	str	d1, [x4, #8]
	b	.LBB21_7
.LBB21_15:                              //   in Loop: Header=BB21_18 Depth=1
	str	d1, [x6, #8]
.LBB21_16:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x6, x1
.LBB21_17:                              //   in Loop: Header=BB21_18 Depth=1
	add	x3, x3, #8
	add	x4, x4, #8
	str	d0, [x6]
	cmp	x3, #128
	b.eq	.LBB21_24
.LBB21_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_23 Depth 2
	mov	x6, x5
	add	x5, x1, x3
	ldr	d1, [x1]
	ldr	d0, [x5]
	fcmp	d0, d1
	b.pl	.LBB21_21
// %bb.19:                              //   in Loop: Header=BB21_18 Depth=1
	cmp	x3, #9
	b.lo	.LBB21_15
// %bb.20:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x0, x2
	mov	x2, x3
	bl	memmove
	b	.LBB21_16
.LBB21_21:                              //   in Loop: Header=BB21_18 Depth=1
	ldr	d1, [x6]
	mov	x6, x5
	fcmp	d0, d1
	b.pl	.LBB21_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB21_18 Depth=1
	mov	x6, x4
.LBB21_23:                              //   Parent Loop BB21_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x6]
	ldur	d1, [x6, #-16]
	sub	x6, x6, #8
	fcmp	d0, d1
	b.mi	.LBB21_23
	b	.LBB21_17
.LBB21_24:
	add	x1, x1, #128
	cmp	x1, x0
	b.ne	.LBB21_27
.LBB21_25:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB21_26:                              //   in Loop: Header=BB21_27 Depth=1
	.cfi_restore_state
	add	x1, x1, #8
	str	d0, [x2]
	cmp	x1, x0
	b.eq	.LBB21_25
.LBB21_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_29 Depth 2
	ldp	d1, d0, [x1, #-8]
	mov	x2, x1
	fcmp	d0, d1
	b.pl	.LBB21_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB21_27 Depth=1
	mov	x2, x1
.LBB21_29:                              //   Parent Loop BB21_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x2]
	ldur	d1, [x2, #-16]
	sub	x2, x2, #8
	fcmp	d0, d1
	b.mi	.LBB21_29
	b	.LBB21_26
.Lfunc_end21:
	.size	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_, .Lfunc_end21-_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_,"axG",@progbits,_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_,comdat
	.weak	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_ // -- Begin function _ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_
	.p2align	2
	.type	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_,@function
_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_: // @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_
	.cfi_startproc
// %bb.0:
	sub	x4, x1, x0
	asr	x1, x4, #3
	subs	x3, x1, #2
	b.ge	.LBB22_2
.LBB22_1:
	ret
.LBB22_2:
	sub	x1, x1, #1
	lsr	x2, x3, #1
	lsr	x1, x1, #1
	tbnz	w4, #3, .LBB22_18
// %bb.3:
	orr	x3, x3, #0x1
	mov	x5, x2
	b	.LBB22_6
.LBB22_4:                               //   in Loop: Header=BB22_6 Depth=1
	mov	x6, x5
.LBB22_5:                               //   in Loop: Header=BB22_6 Depth=1
	sub	x5, x4, #1
	str	d0, [x0, x6, lsl #3]
	cbz	x4, .LBB22_1
.LBB22_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_9 Depth 2
                                        //     Child Loop BB22_14 Depth 2
	ldr	d0, [x0, x5, lsl #3]
	mov	x4, x5
	cmp	x5, x1
	b.ge	.LBB22_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB22_6 Depth=1
	mov	x6, x4
	b	.LBB22_9
.LBB22_8:                               // %select.end
                                        //   in Loop: Header=BB22_9 Depth=2
	ldr	d1, [x0, x5, lsl #3]
	cmp	x5, x1
	str	d1, [x0, x6, lsl #3]
	mov	x6, x5
	b.ge	.LBB22_11
.LBB22_9:                               //   Parent Loop BB22_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x6, #1
	add	x8, x0, x6, lsl #4
	add	x5, x7, #2
	ldr	d2, [x8, #8]
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB22_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB22_9 Depth=2
	orr	x5, x7, #0x1
	b	.LBB22_8
.LBB22_11:                              //   in Loop: Header=BB22_6 Depth=1
	cmp	x5, x2
	b.ne	.LBB22_13
// %bb.12:                              //   in Loop: Header=BB22_6 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	mov	x5, x3
	str	d1, [x0, x2, lsl #3]
.LBB22_13:                              //   in Loop: Header=BB22_6 Depth=1
	cmp	x5, x4
	b.le	.LBB22_4
.LBB22_14:                              //   Parent Loop BB22_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_4
// %bb.15:                              //   in Loop: Header=BB22_14 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.gt	.LBB22_14
	b	.LBB22_5
.LBB22_16:                              //   in Loop: Header=BB22_18 Depth=1
	mov	x4, x2
.LBB22_17:                              //   in Loop: Header=BB22_18 Depth=1
	sub	x2, x3, #1
	str	d0, [x0, x4, lsl #3]
	cbz	x3, .LBB22_1
.LBB22_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_21 Depth 2
                                        //     Child Loop BB22_24 Depth 2
	ldr	d0, [x0, x2, lsl #3]
	mov	x3, x2
	cmp	x2, x1
	mov	x4, x2
	b.ge	.LBB22_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB22_18 Depth=1
	mov	x4, x3
	b	.LBB22_21
.LBB22_20:                              // %select.end10
                                        //   in Loop: Header=BB22_21 Depth=2
	ldr	d1, [x0, x2, lsl #3]
	cmp	x2, x1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x2
	b.ge	.LBB22_23
.LBB22_21:                              //   Parent Loop BB22_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x5, x4, #1
	add	x6, x0, x4, lsl #4
	add	x2, x5, #2
	ldr	d2, [x6, #8]
	ldr	d1, [x0, x2, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB22_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB22_21 Depth=2
	orr	x2, x5, #0x1
	b	.LBB22_20
.LBB22_23:                              //   in Loop: Header=BB22_18 Depth=1
	cmp	x2, x3
	b.le	.LBB22_16
.LBB22_24:                              //   Parent Loop BB22_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x4, x2, #1
	add	x4, x4, x4, lsr #63
	asr	x4, x4, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_16
// %bb.25:                              //   in Loop: Header=BB22_24 Depth=2
	cmp	x4, x3
	str	d1, [x0, x2, lsl #3]
	mov	x2, x4
	b.gt	.LBB22_24
	b	.LBB22_17
.Lfunc_end22:
	.size	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_, .Lfunc_end22-_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_ // -- Begin function _ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	.p2align	2
	.type	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_,@function
_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_: // @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	.cfi_startproc
// %bb.0:
	sub	x6, x1, x0
	cmp	x6, #129
	b.lt	.LBB23_38
// %bb.1:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x5, #-8                         // =0xfffffffffffffff8
	mov	x2, x3
	add	x4, x0, #8
	sub	x5, x5, x0
	b	.LBB23_3
.LBB23_2:                               //   in Loop: Header=BB23_3 Depth=1
	and	x3, x2, #0xff
	mov	x0, x9
	mov	x2, x3
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	cmp	x6, #128
	mov	x1, x9
	b.le	.LBB23_37
.LBB23_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_16 Depth 2
                                        //       Child Loop BB23_17 Depth 3
                                        //       Child Loop BB23_19 Depth 3
	cbz	x3, .LBB23_22
// %bb.4:                               //   in Loop: Header=BB23_3 Depth=1
	lsr	x6, x6, #4
	ldr	d1, [x0, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x0, x6, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB23_7
// %bb.5:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB23_9
// %bb.6:                               //   in Loop: Header=BB23_3 Depth=1
	ldr	d0, [x0]
	str	d2, [x0]
	str	d0, [x0, x6, lsl #3]
	b	.LBB23_15
.LBB23_7:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB23_11
// %bb.8:                               //   in Loop: Header=BB23_3 Depth=1
	ldr	d0, [x0]
	stp	d1, d0, [x0]
	b	.LBB23_15
.LBB23_9:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x0]
	b.pl	.LBB23_13
// %bb.10:                              //   in Loop: Header=BB23_3 Depth=1
	str	d0, [x0]
	stur	d2, [x1, #-8]
	b	.LBB23_15
.LBB23_11:                              //   in Loop: Header=BB23_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x0]
	b.pl	.LBB23_14
// %bb.12:                              //   in Loop: Header=BB23_3 Depth=1
	str	d0, [x0]
	stur	d1, [x1, #-8]
	b	.LBB23_15
.LBB23_13:                              //   in Loop: Header=BB23_3 Depth=1
	stp	d1, d2, [x0]
	b	.LBB23_15
.LBB23_14:                              //   in Loop: Header=BB23_3 Depth=1
	str	d2, [x0]
	str	d1, [x0, x6, lsl #3]
.LBB23_15:                              // %.preheader6
                                        //   in Loop: Header=BB23_3 Depth=1
	sub	x3, x3, #1
	mov	x7, x1
	mov	x8, x4
.LBB23_16:                              //   Parent Loop BB23_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_17 Depth 3
                                        //       Child Loop BB23_19 Depth 3
	ldr	d0, [x0]
	add	x6, x5, x8
.LBB23_17:                              //   Parent Loop BB23_3 Depth=1
                                        //     Parent Loop BB23_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8], #8
	add	x6, x6, #8
	fcmp	d1, d0
	b.mi	.LBB23_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB23_16 Depth=2
	sub	x9, x8, #8
.LBB23_19:                              //   Parent Loop BB23_3 Depth=1
                                        //     Parent Loop BB23_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x7, #-8]!
	fcmp	d0, d2
	b.mi	.LBB23_19
// %bb.20:                              //   in Loop: Header=BB23_16 Depth=2
	cmp	x9, x7
	b.hs	.LBB23_2
// %bb.21:                              //   in Loop: Header=BB23_16 Depth=2
	str	d2, [x9]
	str	d1, [x7]
	b	.LBB23_16
.LBB23_22:
	sturb	w2, [x29, #-4]
	sub	x2, x29, #4
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_
	mov	w2, #1                          // =0x1
	b	.LBB23_25
.LBB23_23:                              //   in Loop: Header=BB23_25 Depth=1
	mov	x4, xzr
.LBB23_24:                              //   in Loop: Header=BB23_25 Depth=1
	cmp	x3, #8
	str	d0, [x0, x4, lsl #3]
	b.le	.LBB23_37
.LBB23_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_28 Depth 2
                                        //     Child Loop BB23_35 Depth 2
	ldr	d0, [x1, #-8]!
	sub	x3, x1, x0
	asr	x5, x3, #3
	ldr	d1, [x0]
	cmp	x5, #3
	str	d1, [x1]
	b.lt	.LBB23_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB23_25 Depth=1
	sub	x4, x5, #1
	mov	x7, xzr
	add	x4, x4, x4, lsr #63
	asr	x6, x4, #1
	b	.LBB23_28
.LBB23_27:                              // %select.end
                                        //   in Loop: Header=BB23_28 Depth=2
	ldr	d1, [x0, x4, lsl #3]
	cmp	x4, x6
	str	d1, [x0, x7, lsl #3]
	mov	x7, x4
	b.ge	.LBB23_31
.LBB23_28:                              //   Parent Loop BB23_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x8, x7, #1
	add	x9, x0, x7, lsl #4
	add	x4, x8, #2
	ldr	d2, [x9, #8]
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB23_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB23_28 Depth=2
	orr	x4, x8, #0x1
	b	.LBB23_27
.LBB23_30:                              //   in Loop: Header=BB23_25 Depth=1
	mov	x4, xzr
.LBB23_31:                              //   in Loop: Header=BB23_25 Depth=1
	tbnz	w3, #3, .LBB23_34
// %bb.32:                              //   in Loop: Header=BB23_25 Depth=1
	sub	x5, x5, #2
	cmp	x4, x5, asr #1
	b.ne	.LBB23_34
// %bb.33:                              //   in Loop: Header=BB23_25 Depth=1
	orr	x5, x2, x4, lsl #1
	ldr	d1, [x0, x5, lsl #3]
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b	.LBB23_35
.LBB23_34:                              //   in Loop: Header=BB23_25 Depth=1
	cbz	x4, .LBB23_24
.LBB23_35:                              //   Parent Loop BB23_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x4, #1
	lsr	x5, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB23_24
// %bb.36:                              //   in Loop: Header=BB23_35 Depth=2
	cmp	x6, #1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b.hi	.LBB23_35
	b	.LBB23_23
.LBB23_37:
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB23_38:
	ret
.Lfunc_end23:
	.size	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_, .Lfunc_end23-_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_ // -- Begin function _ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_
	.p2align	2
	.type	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_,@function
_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_: // @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	sub	x2, x1, x1
	mov	x0, x1
	cmp	x2, #129
	b.lt	.LBB24_2
// %bb.1:
	add	x2, x1, #8
	mov	w3, #8                          // =0x8
	mov	x5, x1
	mov	x4, x2
	b	.LBB24_18
.LBB24_2:
	cmp	x1, x0
	b.eq	.LBB24_25
// %bb.3:
	add	x5, x1, #8
	cmp	x5, x0
	b.eq	.LBB24_25
// %bb.4:                               // %.preheader7
	mov	x2, x1
	b	.LBB24_9
.LBB24_5:                               //   in Loop: Header=BB24_9 Depth=1
	sub	x3, x2, x1
	asr	x5, x3, #3
	cmp	x5, #2
	b.lt	.LBB24_13
// %bb.6:                               //   in Loop: Header=BB24_9 Depth=1
	sub	x4, x4, x5, lsl #3
	mov	x2, x3
	add	x0, x4, #16
	bl	memmove
.LBB24_7:                               //   in Loop: Header=BB24_9 Depth=1
	mov	x3, x1
.LBB24_8:                               //   in Loop: Header=BB24_9 Depth=1
	add	x5, x2, #8
	str	d0, [x3]
	cmp	x5, x0
	b.eq	.LBB24_25
.LBB24_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_12 Depth 2
	ldr	d0, [x2, #8]
	ldr	d1, [x1]
	mov	x4, x2
	mov	x2, x5
	fcmp	d0, d1
	b.mi	.LBB24_5
// %bb.10:                              //   in Loop: Header=BB24_9 Depth=1
	ldr	d1, [x4]
	mov	x3, x2
	fcmp	d0, d1
	b.pl	.LBB24_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB24_9 Depth=1
	mov	x3, x2
.LBB24_12:                              //   Parent Loop BB24_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x3]
	ldur	d1, [x3, #-16]
	sub	x3, x3, #8
	fcmp	d0, d1
	b.mi	.LBB24_12
	b	.LBB24_8
.LBB24_13:                              //   in Loop: Header=BB24_9 Depth=1
	cmp	x3, #8
	mov	x3, x1
	b.ne	.LBB24_8
// %bb.14:                              //   in Loop: Header=BB24_9 Depth=1
	str	d1, [x4, #8]
	b	.LBB24_7
.LBB24_15:                              //   in Loop: Header=BB24_18 Depth=1
	str	d1, [x6, #8]
.LBB24_16:                              //   in Loop: Header=BB24_18 Depth=1
	mov	x6, x1
.LBB24_17:                              //   in Loop: Header=BB24_18 Depth=1
	add	x3, x3, #8
	add	x4, x4, #8
	str	d0, [x6]
	cmp	x3, #128
	b.eq	.LBB24_24
.LBB24_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_23 Depth 2
	mov	x6, x5
	add	x5, x1, x3
	ldr	d1, [x1]
	ldr	d0, [x5]
	fcmp	d0, d1
	b.pl	.LBB24_21
// %bb.19:                              //   in Loop: Header=BB24_18 Depth=1
	cmp	x3, #9
	b.lo	.LBB24_15
// %bb.20:                              //   in Loop: Header=BB24_18 Depth=1
	mov	x0, x2
	mov	x2, x3
	bl	memmove
	b	.LBB24_16
.LBB24_21:                              //   in Loop: Header=BB24_18 Depth=1
	ldr	d1, [x6]
	mov	x6, x5
	fcmp	d0, d1
	b.pl	.LBB24_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB24_18 Depth=1
	mov	x6, x4
.LBB24_23:                              //   Parent Loop BB24_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x6]
	ldur	d1, [x6, #-16]
	sub	x6, x6, #8
	fcmp	d0, d1
	b.mi	.LBB24_23
	b	.LBB24_17
.LBB24_24:
	add	x1, x1, #128
	cmp	x1, x0
	b.ne	.LBB24_27
.LBB24_25:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB24_26:                              //   in Loop: Header=BB24_27 Depth=1
	.cfi_restore_state
	add	x1, x1, #8
	str	d0, [x2]
	cmp	x1, x0
	b.eq	.LBB24_25
.LBB24_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_29 Depth 2
	ldp	d1, d0, [x1, #-8]
	mov	x2, x1
	fcmp	d0, d1
	b.pl	.LBB24_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB24_27 Depth=1
	mov	x2, x1
.LBB24_29:                              //   Parent Loop BB24_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x2]
	ldur	d1, [x2, #-16]
	sub	x2, x2, #8
	fcmp	d0, d1
	b.mi	.LBB24_29
	b	.LBB24_26
.Lfunc_end24:
	.size	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_, .Lfunc_end24-_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_,"axG",@progbits,_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_,comdat
	.weak	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_ // -- Begin function _ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_
	.p2align	2
	.type	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_,@function
_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_: // @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_
	.cfi_startproc
// %bb.0:
	sub	x4, x1, x0
	asr	x1, x4, #3
	subs	x3, x1, #2
	b.ge	.LBB25_2
.LBB25_1:
	ret
.LBB25_2:
	sub	x1, x1, #1
	lsr	x2, x3, #1
	lsr	x1, x1, #1
	tbnz	w4, #3, .LBB25_18
// %bb.3:
	orr	x3, x3, #0x1
	mov	x5, x2
	b	.LBB25_6
.LBB25_4:                               //   in Loop: Header=BB25_6 Depth=1
	mov	x6, x5
.LBB25_5:                               //   in Loop: Header=BB25_6 Depth=1
	sub	x5, x4, #1
	str	d0, [x0, x6, lsl #3]
	cbz	x4, .LBB25_1
.LBB25_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_9 Depth 2
                                        //     Child Loop BB25_14 Depth 2
	ldr	d0, [x0, x5, lsl #3]
	mov	x4, x5
	cmp	x5, x1
	b.ge	.LBB25_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB25_6 Depth=1
	mov	x6, x4
	b	.LBB25_9
.LBB25_8:                               // %select.end
                                        //   in Loop: Header=BB25_9 Depth=2
	ldr	d1, [x0, x5, lsl #3]
	cmp	x5, x1
	str	d1, [x0, x6, lsl #3]
	mov	x6, x5
	b.ge	.LBB25_11
.LBB25_9:                               //   Parent Loop BB25_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x6, #1
	add	x8, x0, x6, lsl #4
	add	x5, x7, #2
	ldr	d2, [x8, #8]
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB25_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB25_9 Depth=2
	orr	x5, x7, #0x1
	b	.LBB25_8
.LBB25_11:                              //   in Loop: Header=BB25_6 Depth=1
	cmp	x5, x2
	b.ne	.LBB25_13
// %bb.12:                              //   in Loop: Header=BB25_6 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	mov	x5, x3
	str	d1, [x0, x2, lsl #3]
.LBB25_13:                              //   in Loop: Header=BB25_6 Depth=1
	cmp	x5, x4
	b.le	.LBB25_4
.LBB25_14:                              //   Parent Loop BB25_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB25_4
// %bb.15:                              //   in Loop: Header=BB25_14 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.gt	.LBB25_14
	b	.LBB25_5
.LBB25_16:                              //   in Loop: Header=BB25_18 Depth=1
	mov	x4, x2
.LBB25_17:                              //   in Loop: Header=BB25_18 Depth=1
	sub	x2, x3, #1
	str	d0, [x0, x4, lsl #3]
	cbz	x3, .LBB25_1
.LBB25_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_21 Depth 2
                                        //     Child Loop BB25_24 Depth 2
	ldr	d0, [x0, x2, lsl #3]
	mov	x3, x2
	cmp	x2, x1
	mov	x4, x2
	b.ge	.LBB25_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB25_18 Depth=1
	mov	x4, x3
	b	.LBB25_21
.LBB25_20:                              // %select.end10
                                        //   in Loop: Header=BB25_21 Depth=2
	ldr	d1, [x0, x2, lsl #3]
	cmp	x2, x1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x2
	b.ge	.LBB25_23
.LBB25_21:                              //   Parent Loop BB25_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x5, x4, #1
	add	x6, x0, x4, lsl #4
	add	x2, x5, #2
	ldr	d2, [x6, #8]
	ldr	d1, [x0, x2, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB25_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB25_21 Depth=2
	orr	x2, x5, #0x1
	b	.LBB25_20
.LBB25_23:                              //   in Loop: Header=BB25_18 Depth=1
	cmp	x2, x3
	b.le	.LBB25_16
.LBB25_24:                              //   Parent Loop BB25_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x4, x2, #1
	add	x4, x4, x4, lsr #63
	asr	x4, x4, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB25_16
// %bb.25:                              //   in Loop: Header=BB25_24 Depth=2
	cmp	x4, x3
	str	d1, [x0, x2, lsl #3]
	mov	x2, x4
	b.gt	.LBB25_24
	b	.LBB25_17
.Lfunc_end25:
	.size	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_, .Lfunc_end25-_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_,"axG",@progbits,_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_,comdat
	.weak	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_ // -- Begin function _ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	.p2align	2
	.type	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_,@function
_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_: // @_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	.cfi_startproc
// %bb.0:
	sub	x5, x1, x0
	cmp	x5, #129
	b.lt	.LBB26_38
// %bb.1:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x4, #-8                         // =0xfffffffffffffff8
	add	x3, x0, #8
	sub	x4, x4, x0
	b	.LBB26_3
.LBB26_2:                               //   in Loop: Header=BB26_3 Depth=1
	mov	x0, x8
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	cmp	x5, #128
	mov	x1, x8
	b.le	.LBB26_37
.LBB26_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_16 Depth 2
                                        //       Child Loop BB26_17 Depth 3
                                        //       Child Loop BB26_19 Depth 3
	cbz	x2, .LBB26_22
// %bb.4:                               //   in Loop: Header=BB26_3 Depth=1
	lsr	x5, x5, #4
	ldr	d1, [x0, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x0, x5, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB26_7
// %bb.5:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB26_9
// %bb.6:                               //   in Loop: Header=BB26_3 Depth=1
	ldr	d0, [x0]
	str	d2, [x0]
	str	d0, [x0, x5, lsl #3]
	b	.LBB26_15
.LBB26_7:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB26_11
// %bb.8:                               //   in Loop: Header=BB26_3 Depth=1
	ldr	d0, [x0]
	stp	d1, d0, [x0]
	b	.LBB26_15
.LBB26_9:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x0]
	b.pl	.LBB26_13
// %bb.10:                              //   in Loop: Header=BB26_3 Depth=1
	str	d0, [x0]
	stur	d2, [x1, #-8]
	b	.LBB26_15
.LBB26_11:                              //   in Loop: Header=BB26_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x0]
	b.pl	.LBB26_14
// %bb.12:                              //   in Loop: Header=BB26_3 Depth=1
	str	d0, [x0]
	stur	d1, [x1, #-8]
	b	.LBB26_15
.LBB26_13:                              //   in Loop: Header=BB26_3 Depth=1
	stp	d1, d2, [x0]
	b	.LBB26_15
.LBB26_14:                              //   in Loop: Header=BB26_3 Depth=1
	str	d2, [x0]
	str	d1, [x0, x5, lsl #3]
.LBB26_15:                              // %.preheader6
                                        //   in Loop: Header=BB26_3 Depth=1
	sub	x2, x2, #1
	mov	x6, x1
	mov	x7, x3
.LBB26_16:                              //   Parent Loop BB26_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB26_17 Depth 3
                                        //       Child Loop BB26_19 Depth 3
	ldr	d0, [x0]
	add	x5, x4, x7
.LBB26_17:                              //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x7], #8
	add	x5, x5, #8
	fcmp	d1, d0
	b.mi	.LBB26_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB26_16 Depth=2
	sub	x8, x7, #8
.LBB26_19:                              //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x6, #-8]!
	fcmp	d0, d2
	b.mi	.LBB26_19
// %bb.20:                              //   in Loop: Header=BB26_16 Depth=2
	cmp	x8, x6
	b.hs	.LBB26_2
// %bb.21:                              //   in Loop: Header=BB26_16 Depth=2
	str	d2, [x8]
	str	d1, [x6]
	b	.LBB26_16
.LBB26_22:
	sub	x2, x29, #1
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_
	mov	w2, #1                          // =0x1
	b	.LBB26_25
.LBB26_23:                              //   in Loop: Header=BB26_25 Depth=1
	mov	x4, xzr
.LBB26_24:                              //   in Loop: Header=BB26_25 Depth=1
	cmp	x3, #8
	str	d0, [x0, x4, lsl #3]
	b.le	.LBB26_37
.LBB26_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_28 Depth 2
                                        //     Child Loop BB26_35 Depth 2
	ldr	d0, [x1, #-8]!
	sub	x3, x1, x0
	asr	x5, x3, #3
	ldr	d1, [x0]
	cmp	x5, #3
	str	d1, [x1]
	b.lt	.LBB26_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB26_25 Depth=1
	sub	x4, x5, #1
	mov	x7, xzr
	add	x4, x4, x4, lsr #63
	asr	x6, x4, #1
	b	.LBB26_28
.LBB26_27:                              // %select.end
                                        //   in Loop: Header=BB26_28 Depth=2
	ldr	d1, [x0, x4, lsl #3]
	cmp	x4, x6
	str	d1, [x0, x7, lsl #3]
	mov	x7, x4
	b.ge	.LBB26_31
.LBB26_28:                              //   Parent Loop BB26_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x8, x7, #1
	add	x9, x0, x7, lsl #4
	add	x4, x8, #2
	ldr	d2, [x9, #8]
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB26_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB26_28 Depth=2
	orr	x4, x8, #0x1
	b	.LBB26_27
.LBB26_30:                              //   in Loop: Header=BB26_25 Depth=1
	mov	x4, xzr
.LBB26_31:                              //   in Loop: Header=BB26_25 Depth=1
	tbnz	w3, #3, .LBB26_34
// %bb.32:                              //   in Loop: Header=BB26_25 Depth=1
	sub	x5, x5, #2
	cmp	x4, x5, asr #1
	b.ne	.LBB26_34
// %bb.33:                              //   in Loop: Header=BB26_25 Depth=1
	orr	x5, x2, x4, lsl #1
	ldr	d1, [x0, x5, lsl #3]
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b	.LBB26_35
.LBB26_34:                              //   in Loop: Header=BB26_25 Depth=1
	cbz	x4, .LBB26_24
.LBB26_35:                              //   Parent Loop BB26_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x4, #1
	lsr	x5, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB26_24
// %bb.36:                              //   in Loop: Header=BB26_35 Depth=2
	cmp	x6, #1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x5
	b.hi	.LBB26_35
	b	.LBB26_23
.LBB26_37:
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB26_38:
	ret
.Lfunc_end26:
	.size	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_, .Lfunc_end26-_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_,"axG",@progbits,_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_,comdat
	.weak	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_ // -- Begin function _ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_
	.p2align	2
	.type	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_,@function
_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_: // @_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	.cfi_remember_state
	sub	x2, x1, x1
	mov	x0, x1
	cmp	x2, #129
	b.lt	.LBB27_2
// %bb.1:
	add	x2, x1, #8
	mov	w3, #8                          // =0x8
	mov	x5, x1
	mov	x4, x2
	b	.LBB27_18
.LBB27_2:
	cmp	x1, x0
	b.eq	.LBB27_25
// %bb.3:
	add	x5, x1, #8
	cmp	x5, x0
	b.eq	.LBB27_25
// %bb.4:                               // %.preheader7
	mov	x2, x1
	b	.LBB27_9
.LBB27_5:                               //   in Loop: Header=BB27_9 Depth=1
	sub	x3, x2, x1
	asr	x5, x3, #3
	cmp	x5, #2
	b.lt	.LBB27_13
// %bb.6:                               //   in Loop: Header=BB27_9 Depth=1
	sub	x4, x4, x5, lsl #3
	mov	x2, x3
	add	x0, x4, #16
	bl	memmove
.LBB27_7:                               //   in Loop: Header=BB27_9 Depth=1
	mov	x3, x1
.LBB27_8:                               //   in Loop: Header=BB27_9 Depth=1
	add	x5, x2, #8
	str	d0, [x3]
	cmp	x5, x0
	b.eq	.LBB27_25
.LBB27_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_12 Depth 2
	ldr	d0, [x2, #8]
	ldr	d1, [x1]
	mov	x4, x2
	mov	x2, x5
	fcmp	d0, d1
	b.mi	.LBB27_5
// %bb.10:                              //   in Loop: Header=BB27_9 Depth=1
	ldr	d1, [x4]
	mov	x3, x2
	fcmp	d0, d1
	b.pl	.LBB27_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB27_9 Depth=1
	mov	x3, x2
.LBB27_12:                              //   Parent Loop BB27_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x3]
	ldur	d1, [x3, #-16]
	sub	x3, x3, #8
	fcmp	d0, d1
	b.mi	.LBB27_12
	b	.LBB27_8
.LBB27_13:                              //   in Loop: Header=BB27_9 Depth=1
	cmp	x3, #8
	mov	x3, x1
	b.ne	.LBB27_8
// %bb.14:                              //   in Loop: Header=BB27_9 Depth=1
	str	d1, [x4, #8]
	b	.LBB27_7
.LBB27_15:                              //   in Loop: Header=BB27_18 Depth=1
	str	d1, [x6, #8]
.LBB27_16:                              //   in Loop: Header=BB27_18 Depth=1
	mov	x6, x1
.LBB27_17:                              //   in Loop: Header=BB27_18 Depth=1
	add	x3, x3, #8
	add	x4, x4, #8
	str	d0, [x6]
	cmp	x3, #128
	b.eq	.LBB27_24
.LBB27_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_23 Depth 2
	mov	x6, x5
	add	x5, x1, x3
	ldr	d1, [x1]
	ldr	d0, [x5]
	fcmp	d0, d1
	b.pl	.LBB27_21
// %bb.19:                              //   in Loop: Header=BB27_18 Depth=1
	cmp	x3, #9
	b.lo	.LBB27_15
// %bb.20:                              //   in Loop: Header=BB27_18 Depth=1
	mov	x0, x2
	mov	x2, x3
	bl	memmove
	b	.LBB27_16
.LBB27_21:                              //   in Loop: Header=BB27_18 Depth=1
	ldr	d1, [x6]
	mov	x6, x5
	fcmp	d0, d1
	b.pl	.LBB27_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB27_18 Depth=1
	mov	x6, x4
.LBB27_23:                              //   Parent Loop BB27_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x6]
	ldur	d1, [x6, #-16]
	sub	x6, x6, #8
	fcmp	d0, d1
	b.mi	.LBB27_23
	b	.LBB27_17
.LBB27_24:
	add	x1, x1, #128
	cmp	x1, x0
	b.ne	.LBB27_27
.LBB27_25:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.LBB27_26:                              //   in Loop: Header=BB27_27 Depth=1
	.cfi_restore_state
	add	x1, x1, #8
	str	d0, [x2]
	cmp	x1, x0
	b.eq	.LBB27_25
.LBB27_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_29 Depth 2
	ldp	d1, d0, [x1, #-8]
	mov	x2, x1
	fcmp	d0, d1
	b.pl	.LBB27_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB27_27 Depth=1
	mov	x2, x1
.LBB27_29:                              //   Parent Loop BB27_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x2]
	ldur	d1, [x2, #-16]
	sub	x2, x2, #8
	fcmp	d0, d1
	b.mi	.LBB27_29
	b	.LBB27_26
.Lfunc_end27:
	.size	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_, .Lfunc_end27-_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_,"axG",@progbits,_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_,comdat
	.weak	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_ // -- Begin function _ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_
	.p2align	2
	.type	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_,@function
_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_: // @_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_
	.cfi_startproc
// %bb.0:
	sub	x4, x1, x0
	asr	x1, x4, #3
	subs	x3, x1, #2
	b.ge	.LBB28_2
.LBB28_1:
	ret
.LBB28_2:
	sub	x1, x1, #1
	lsr	x2, x3, #1
	lsr	x1, x1, #1
	tbnz	w4, #3, .LBB28_18
// %bb.3:
	orr	x3, x3, #0x1
	mov	x5, x2
	b	.LBB28_6
.LBB28_4:                               //   in Loop: Header=BB28_6 Depth=1
	mov	x6, x5
.LBB28_5:                               //   in Loop: Header=BB28_6 Depth=1
	sub	x5, x4, #1
	str	d0, [x0, x6, lsl #3]
	cbz	x4, .LBB28_1
.LBB28_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_9 Depth 2
                                        //     Child Loop BB28_14 Depth 2
	ldr	d0, [x0, x5, lsl #3]
	mov	x4, x5
	cmp	x5, x1
	b.ge	.LBB28_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB28_6 Depth=1
	mov	x6, x4
	b	.LBB28_9
.LBB28_8:                               // %select.end
                                        //   in Loop: Header=BB28_9 Depth=2
	ldr	d1, [x0, x5, lsl #3]
	cmp	x5, x1
	str	d1, [x0, x6, lsl #3]
	mov	x6, x5
	b.ge	.LBB28_11
.LBB28_9:                               //   Parent Loop BB28_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x7, x6, #1
	add	x8, x0, x6, lsl #4
	add	x5, x7, #2
	ldr	d2, [x8, #8]
	ldr	d1, [x0, x5, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB28_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB28_9 Depth=2
	orr	x5, x7, #0x1
	b	.LBB28_8
.LBB28_11:                              //   in Loop: Header=BB28_6 Depth=1
	cmp	x5, x2
	b.ne	.LBB28_13
// %bb.12:                              //   in Loop: Header=BB28_6 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	mov	x5, x3
	str	d1, [x0, x2, lsl #3]
.LBB28_13:                              //   in Loop: Header=BB28_6 Depth=1
	cmp	x5, x4
	b.le	.LBB28_4
.LBB28_14:                              //   Parent Loop BB28_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB28_4
// %bb.15:                              //   in Loop: Header=BB28_14 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.gt	.LBB28_14
	b	.LBB28_5
.LBB28_16:                              //   in Loop: Header=BB28_18 Depth=1
	mov	x4, x2
.LBB28_17:                              //   in Loop: Header=BB28_18 Depth=1
	sub	x2, x3, #1
	str	d0, [x0, x4, lsl #3]
	cbz	x3, .LBB28_1
.LBB28_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_21 Depth 2
                                        //     Child Loop BB28_24 Depth 2
	ldr	d0, [x0, x2, lsl #3]
	mov	x3, x2
	cmp	x2, x1
	mov	x4, x2
	b.ge	.LBB28_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB28_18 Depth=1
	mov	x4, x3
	b	.LBB28_21
.LBB28_20:                              // %select.end10
                                        //   in Loop: Header=BB28_21 Depth=2
	ldr	d1, [x0, x2, lsl #3]
	cmp	x2, x1
	str	d1, [x0, x4, lsl #3]
	mov	x4, x2
	b.ge	.LBB28_23
.LBB28_21:                              //   Parent Loop BB28_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x5, x4, #1
	add	x6, x0, x4, lsl #4
	add	x2, x5, #2
	ldr	d2, [x6, #8]
	ldr	d1, [x0, x2, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB28_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB28_21 Depth=2
	orr	x2, x5, #0x1
	b	.LBB28_20
.LBB28_23:                              //   in Loop: Header=BB28_18 Depth=1
	cmp	x2, x3
	b.le	.LBB28_16
.LBB28_24:                              //   Parent Loop BB28_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x4, x2, #1
	add	x4, x4, x4, lsr #63
	asr	x4, x4, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB28_16
// %bb.25:                              //   in Loop: Header=BB28_24 Depth=2
	cmp	x4, x3
	str	d1, [x0, x2, lsl #3]
	mov	x2, x4
	b.gt	.LBB28_24
	b	.LBB28_17
.Lfunc_end28:
	.size	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_, .Lfunc_end28-_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_
	.cfi_endproc
                                        // -- End function
	.type	results,@object                 // @results
	.bss
	.globl	results
	.p2align	3, 0x0
results:
	.xword	0
	.size	results, 8

	.type	current_test,@object            // @current_test
	.globl	current_test
	.p2align	2, 0x0
current_test:
	.word	0                               // 0x0
	.size	current_test, 4

	.type	allocated_results,@object       // @allocated_results
	.globl	allocated_results
	.p2align	2, 0x0
allocated_results:
	.word	0                               // 0x0
	.size	allocated_results, 4

	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"Could not allocate %d results\n"
	.size	.L.str, 31

	.type	.L.str.1,@object                // @.str.1
.L.str.1:
	.asciz	"\ntest %*s description   absolute   operations   ratio with\n"
	.size	.L.str.1, 60

	.type	.L.str.2,@object                // @.str.2
.L.str.2:
	.asciz	" "
	.size	.L.str.2, 2

	.type	.L.str.3,@object                // @.str.3
.L.str.3:
	.asciz	"number %*s time       per second   test0\n\n"
	.size	.L.str.3, 43

	.type	.L.str.4,@object                // @.str.4
.L.str.4:
	.asciz	"%2i %*s\"%s\"  %5.2f sec   %5.2f M     %.2f\n"
	.size	.L.str.4, 43

	.type	.L.str.5,@object                // @.str.5
.L.str.5:
	.zero	1
	.size	.L.str.5, 1

	.type	.L.str.6,@object                // @.str.6
.L.str.6:
	.asciz	"\nTotal absolute time for %s: %.2f sec\n"
	.size	.L.str.6, 39

	.type	.L.str.7,@object                // @.str.7
.L.str.7:
	.asciz	"\n%s Penalty: %.2f\n\n"
	.size	.L.str.7, 20

	.type	.L.str.8,@object                // @.str.8
.L.str.8:
	.asciz	"\ntest %*s description   absolute\n"
	.size	.L.str.8, 34

	.type	.L.str.9,@object                // @.str.9
.L.str.9:
	.asciz	"number %*s time\n\n"
	.size	.L.str.9, 18

	.type	.L.str.10,@object               // @.str.10
.L.str.10:
	.asciz	"%2i %*s\"%s\"  %5.2f sec\n"
	.size	.L.str.10, 24

	.type	start_time,@object              // @start_time
	.bss
	.globl	start_time
	.p2align	3, 0x0
start_time:
	.xword	0                               // 0x0
	.size	start_time, 8

	.type	end_time,@object                // @end_time
	.globl	end_time
	.p2align	3, 0x0
end_time:
	.xword	0                               // 0x0
	.size	end_time, 8

	.type	.L.str.11,@object               // @.str.11
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.11:
	.asciz	"test %i failed\n"
	.size	.L.str.11, 16

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
