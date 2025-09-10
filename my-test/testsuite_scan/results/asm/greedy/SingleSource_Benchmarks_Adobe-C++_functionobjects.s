	.file	"functionobjects.cpp"
	.text
	.globl	_Z13record_resultdPKc           // -- Begin function _Z13record_resultdPKc
	.p2align	2
	.type	_Z13record_resultdPKc,@function
_Z13record_resultdPKc:                  // @_Z13record_resultdPKc
	.cfi_startproc
// %bb.0:
	str	d8, [sp, #-64]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #32]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	.cfi_offset b8, -64
	.cfi_remember_state
	fmov	d8, d0
	adrp	x22, results
	adrp	x21, allocated_results
	mov	x19, x0
	ldr	x0, [x22, :lo12:results]
	ldr	w9, [x21, :lo12:allocated_results]
	adrp	x20, current_test
	cbz	x0, .LBB0_2
// %bb.1:
	ldr	w8, [x20, :lo12:current_test]
	cmp	w8, w9
	b.lt	.LBB0_4
.LBB0_2:
	add	w8, w9, #10
	sbfiz	x1, x8, #4, #32
	str	w8, [x21, :lo12:allocated_results]
	bl	realloc
	str	x0, [x22, :lo12:results]
	cbz	x0, .LBB0_5
// %bb.3:
	ldr	w8, [x20, :lo12:current_test]
.LBB0_4:
	add	x9, x0, w8, sxtw #4
	add	w8, w8, #1
	str	d8, [x9]
	str	x19, [x9, #8]
	str	w8, [x20, :lo12:current_test]
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #64                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB0_5:
	.cfi_restore_state
	ldr	w1, [x21, :lo12:allocated_results]
	adrp	x0, .L.str
	add	x0, x0, :lo12:.L.str
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
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
	stp	d9, d8, [sp, #16]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #48]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #64]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #80]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #96]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #112]            // 16-byte Folded Spill
	add	x29, sp, #32
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
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	adrp	x26, current_test
	mov	w20, w4
	mov	w22, w2
	ldr	w24, [x26, :lo12:current_test]
	mov	w23, w1
	mov	x19, x0
	adrp	x27, results
	cmp	w24, #1
	b.lt	.LBB1_3
// %bb.1:
	ldr	x8, [x27, :lo12:results]
	mov	w21, #12                        // =0xc
	add	x25, x8, #8
.LBB1_2:                                // =>This Inner Loop Header: Depth=1
	ldr	x0, [x25], #16
	bl	strlen
	cmp	w21, w0
	csel	w21, w21, w0, gt
	subs	x24, x24, #1
	b.ne	.LBB1_2
	b	.LBB1_4
.LBB1_3:
	mov	w21, #12                        // =0xc
.LBB1_4:
	adrp	x24, .L.str.2
	add	x24, x24, :lo12:.L.str.2
	sub	w1, w21, #12
	adrp	x0, .L.str.1
	add	x0, x0, :lo12:.L.str.1
	mov	x2, x24
	bl	printf
	adrp	x0, .L.str.3
	add	x0, x0, :lo12:.L.str.3
	mov	w1, w21
	mov	x2, x24
	bl	printf
	movi	d8, #0000000000000000
	ldr	w8, [x26, :lo12:current_test]
	cmp	w8, #1
	b.lt	.LBB1_15
// %bb.5:                               // %.preheader2
	scvtf	d0, w23
	scvtf	d1, w22
	mov	x8, #145685290680320            // =0x848000000000
	movk	x8, #16686, lsl #48
	mov	x28, xzr
	mov	x24, xzr
	adrp	x22, .L.str.4
	add	x22, x22, :lo12:.L.str.4
	adrp	x23, .L.str.5
	add	x23, x23, :lo12:.L.str.5
	str	w20, [sp, #4]                   // 4-byte Folded Spill
	fmul	d0, d0, d1
	fmov	d1, x8
	str	x19, [sp, #8]                   // 8-byte Folded Spill
	fdiv	d9, d0, d1
.LBB1_6:                                // =>This Inner Loop Header: Depth=1
	ldr	x20, [x27, :lo12:results]
	add	x19, x20, x28
	ldr	x25, [x19, #8]
	mov	x0, x25
	bl	strlen
	ldr	d0, [x19]
	ldr	d2, [x20]
	sub	w2, w21, w0
	mov	x0, x22
	mov	w1, w24
	mov	x3, x23
	fdiv	d1, d9, d0
	mov	x4, x25
	fdiv	d2, d0, d2
	bl	printf
	ldrsw	x8, [x26, :lo12:current_test]
	add	x24, x24, #1
	add	x28, x28, #16
	cmp	x24, x8
	b.lt	.LBB1_6
// %bb.7:
	ldr	x19, [sp, #8]                   // 8-byte Folded Reload
	ldr	w20, [sp, #4]                   // 4-byte Folded Reload
	cmp	w8, #1
	b.lt	.LBB1_15
// %bb.8:
	movi	d8, #0000000000000000
	ldr	x9, [x27, :lo12:results]
	cmp	w8, #1
	b.ne	.LBB1_10
// %bb.9:
	mov	x10, xzr
	b	.LBB1_13
.LBB1_10:
	and	x10, x8, #0x7ffffffe
	add	x11, x9, #16
	mov	x12, x10
.LBB1_11:                               // =>This Inner Loop Header: Depth=1
	ldur	d0, [x11, #-16]
	ldr	d1, [x11], #32
	subs	x12, x12, #2
	fadd	d0, d8, d0
	fadd	d8, d0, d1
	b.ne	.LBB1_11
// %bb.12:
	cmp	x10, x8
	b.eq	.LBB1_15
.LBB1_13:                               // %.preheader
	add	x9, x9, x10, lsl #4
	sub	x8, x8, x10
.LBB1_14:                               // =>This Inner Loop Header: Depth=1
	ldr	d0, [x9], #16
	subs	x8, x8, #1
	fadd	d8, d8, d0
	b.ne	.LBB1_14
.LBB1_15:
	fmov	d0, d8
	adrp	x0, .L.str.6
	add	x0, x0, :lo12:.L.str.6
	mov	x1, x19
	bl	printf
	cbz	w20, .LBB1_20
// %bb.16:
	ldr	w8, [x26, :lo12:current_test]
	cmp	w8, #2
	b.lt	.LBB1_20
// %bb.17:
	ldr	x20, [x27, :lo12:results]
	movi	d8, #0000000000000000
	mov	w21, #1                         // =0x1
	ldr	d9, [x20], #16
.LBB1_18:                               // =>This Inner Loop Header: Depth=1
	ldr	d0, [x20], #16
	fdiv	d0, d0, d9
	bl	log
	fadd	d8, d8, d0
	ldrsw	x8, [x26, :lo12:current_test]
	add	x21, x21, #1
	cmp	x21, x8
	b.lt	.LBB1_18
// %bb.19:
	sub	w8, w8, #1
	scvtf	d0, w8
	fdiv	d0, d8, d0
	bl	exp
	adrp	x0, .L.str.7
	add	x0, x0, :lo12:.L.str.7
	mov	x1, x19
	bl	printf
.LBB1_20:
	str	wzr, [x26, :lo12:current_test]
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #16]               // 16-byte Folded Reload
	add	sp, sp, #128
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
	.cfi_restore b9
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
	str	d8, [sp, #-112]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 112
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #32]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             // 16-byte Folded Spill
	add	x29, sp, #16
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
	adrp	x26, current_test
	mov	x19, x1
	mov	x20, x0
	ldr	w22, [x26, :lo12:current_test]
	adrp	x27, results
	cmp	w22, #1
	b.lt	.LBB2_3
// %bb.1:
	ldr	x8, [x27, :lo12:results]
	mov	w21, #12                        // =0xc
	add	x23, x8, #8
.LBB2_2:                                // =>This Inner Loop Header: Depth=1
	ldr	x0, [x23], #16
	bl	strlen
	cmp	w21, w0
	csel	w21, w21, w0, gt
	subs	x22, x22, #1
	b.ne	.LBB2_2
	b	.LBB2_4
.LBB2_3:
	mov	w21, #12                        // =0xc
.LBB2_4:
	adrp	x22, .L.str.2
	add	x22, x22, :lo12:.L.str.2
	sub	w2, w21, #12
	adrp	x1, .L.str.8
	add	x1, x1, :lo12:.L.str.8
	mov	x0, x20
	mov	x3, x22
	bl	fprintf
	adrp	x1, .L.str.9
	add	x1, x1, :lo12:.L.str.9
	mov	x0, x20
	mov	w2, w21
	mov	x3, x22
	bl	fprintf
	movi	d8, #0000000000000000
	ldr	w8, [x26, :lo12:current_test]
	cmp	w8, #1
	b.lt	.LBB2_15
// %bb.5:                               // %.preheader2
	mov	x28, xzr
	mov	x22, xzr
	adrp	x23, .L.str.10
	add	x23, x23, :lo12:.L.str.10
	adrp	x24, .L.str.5
	add	x24, x24, :lo12:.L.str.5
	str	x19, [sp, #8]                   // 8-byte Folded Spill
.LBB2_6:                                // =>This Inner Loop Header: Depth=1
	ldr	x8, [x27, :lo12:results]
	add	x19, x8, x28
	ldr	x25, [x19, #8]
	mov	x0, x25
	bl	strlen
	ldr	d0, [x19]
	sub	w3, w21, w0
	mov	x0, x20
	mov	x1, x23
	mov	w2, w22
	mov	x4, x24
	mov	x5, x25
	bl	fprintf
	ldrsw	x8, [x26, :lo12:current_test]
	add	x22, x22, #1
	add	x28, x28, #16
	cmp	x22, x8
	b.lt	.LBB2_6
// %bb.7:
	ldr	x19, [sp, #8]                   // 8-byte Folded Reload
	cmp	w8, #1
	b.lt	.LBB2_15
// %bb.8:
	movi	d8, #0000000000000000
	ldr	x9, [x27, :lo12:results]
	cmp	w8, #1
	b.ne	.LBB2_10
// %bb.9:
	mov	x10, xzr
	b	.LBB2_13
.LBB2_10:
	and	x10, x8, #0x7ffffffe
	add	x11, x9, #16
	mov	x12, x10
.LBB2_11:                               // =>This Inner Loop Header: Depth=1
	ldur	d0, [x11, #-16]
	ldr	d1, [x11], #32
	subs	x12, x12, #2
	fadd	d0, d8, d0
	fadd	d8, d0, d1
	b.ne	.LBB2_11
// %bb.12:
	cmp	x10, x8
	b.eq	.LBB2_15
.LBB2_13:                               // %.preheader
	add	x9, x9, x10, lsl #4
	sub	x8, x8, x10
.LBB2_14:                               // =>This Inner Loop Header: Depth=1
	ldr	d0, [x9], #16
	subs	x8, x8, #1
	fadd	d8, d8, d0
	b.ne	.LBB2_14
.LBB2_15:
	fmov	d0, d8
	adrp	x1, .L.str.6
	add	x1, x1, :lo12:.L.str.6
	mov	x0, x20
	mov	x2, x19
	bl	fprintf
	str	wzr, [x26, :lo12:current_test]
	.cfi_def_cfa wsp, 112
	ldp	x20, x19, [sp, #96]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #112                  // 8-byte Folded Reload
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
	adrp	x8, start_time
	str	x0, [x8, :lo12:start_time]
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
	adrp	x8, start_time
	ldr	x8, [x8, :lo12:start_time]
	sub	x8, x0, x8
	scvtf	d0, x8
	mov	x8, #145685290680320            // =0x848000000000
	movk	x8, #16686, lsl #48
	fmov	d1, x8
	adrp	x8, end_time
	fdiv	d0, d0, d1
	str	x0, [x8, :lo12:end_time]
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
	ldr	d0, [x0]
	ldr	d1, [x1]
	fcmp	d0, d1
	cset	w8, gt
	csinv	w0, w8, wzr, pl
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
	fcmp	d0, d1
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
	str	d8, [sp, #-64]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #8]              // 16-byte Folded Spill
	str	x23, [sp, #24]                  // 8-byte Folded Spill
	stp	x22, x21, [sp, #32]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             // 16-byte Folded Spill
	add	x29, sp, #8
	.cfi_def_cfa w29, 56
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w30, -48
	.cfi_offset w29, -56
	.cfi_offset b8, -64
	.cfi_remember_state
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB8_8
// %bb.1:
	ldr	d8, [x0]
	mov	x19, x2
	mov	x20, x1
	mov	x22, x1
	mov	x21, x0
	mov	x23, x0
.LBB8_2:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_5 Depth 2
	fmov	d0, d8
	ldr	d1, [x22, #-8]!
	blr	x19
	tbnz	w0, #0, .LBB8_2
// %bb.3:                               //   in Loop: Header=BB8_2 Depth=1
	cmp	x23, x22
	b.hs	.LBB8_9
// %bb.4:                               // %.preheader
                                        //   in Loop: Header=BB8_2 Depth=1
	sub	x23, x23, #8
.LBB8_5:                                //   Parent Loop BB8_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	fmov	d1, d8
	ldr	d0, [x23, #8]!
	blr	x19
	tbnz	w0, #0, .LBB8_5
// %bb.6:                               //   in Loop: Header=BB8_2 Depth=1
	cmp	x23, x22
	b.hs	.LBB8_9
// %bb.7:                               //   in Loop: Header=BB8_2 Depth=1
	ldr	d0, [x23]
	ldr	d1, [x22]
	str	d0, [x22]
	str	d1, [x23]
	b	.LBB8_2
.LBB8_8:
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldr	x23, [sp, #24]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #8]              // 16-byte Folded Reload
	ldr	d8, [sp], #64                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB8_9:
	.cfi_restore_state
	add	x1, x22, #8
	mov	x0, x21
	mov	x2, x19
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	add	x0, x22, #8
	mov	x1, x20
	mov	x2, x19
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldr	x23, [sp, #24]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #8]              // 16-byte Folded Reload
	ldr	d8, [sp], #64                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
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
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB9_12
// %bb.1:                               // %.preheader1
	str	d8, [sp, #-80]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #8]              // 16-byte Folded Spill
	str	x25, [sp, #24]                  // 8-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #8
	.cfi_def_cfa w29, 72
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w30, -64
	.cfi_offset w29, -72
	.cfi_offset b8, -80
	mov	x19, x1
	mov	x21, x0
	mov	x20, x2
	sub	x23, x1, #8
	b	.LBB9_3
.LBB9_2:                                //   in Loop: Header=BB9_3 Depth=1
	add	x22, x22, #8
	mov	x0, x21
	mov	x2, x20
	mov	x1, x22
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	cmp	x25, #8
	mov	x21, x22
	b.le	.LBB9_11
.LBB9_3:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_4 Depth 2
                                        //       Child Loop BB9_5 Depth 3
                                        //       Child Loop BB9_8 Depth 3
	ldr	d8, [x21]
	mov	x22, x19
	mov	x24, x21
.LBB9_4:                                //   Parent Loop BB9_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB9_5 Depth 3
                                        //       Child Loop BB9_8 Depth 3
	sub	x25, x23, x22
.LBB9_5:                                //   Parent Loop BB9_3 Depth=1
                                        //     Parent Loop BB9_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	fmov	d0, d8
	ldr	d1, [x22, #-8]!
	blr	x20
	add	x25, x25, #8
	tbnz	w0, #0, .LBB9_5
// %bb.6:                               //   in Loop: Header=BB9_4 Depth=2
	cmp	x24, x22
	b.hs	.LBB9_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB9_4 Depth=2
	sub	x24, x24, #8
.LBB9_8:                                //   Parent Loop BB9_3 Depth=1
                                        //     Parent Loop BB9_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	fmov	d1, d8
	ldr	d0, [x24, #8]!
	blr	x20
	tbnz	w0, #0, .LBB9_8
// %bb.9:                               //   in Loop: Header=BB9_4 Depth=2
	cmp	x24, x22
	b.hs	.LBB9_2
// %bb.10:                              //   in Loop: Header=BB9_4 Depth=2
	ldr	d0, [x24]
	ldr	d1, [x22]
	str	d0, [x22]
	str	d1, [x24]
	b	.LBB9_4
.LBB9_11:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldr	x25, [sp, #24]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #8]              // 16-byte Folded Reload
	ldr	d8, [sp], #80                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
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
	sub	sp, sp, #144
	.cfi_def_cfa_offset 144
	str	d8, [sp, #32]                   // 8-byte Folded Spill
	stp	x29, x30, [sp, #48]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #64]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #80]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #96]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #112]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #128]            // 16-byte Folded Spill
	add	x29, sp, #48
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
	cmp	w0, #2
	b.lt	.LBB10_3
// %bb.1:
	mov	w19, w0
	ldr	x0, [x1, #8]
	mov	x20, x1
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	cmp	w19, #2
	mov	x23, x0
	b.ne	.LBB10_8
// %bb.2:
	mov	w19, #10000                     // =0x2710
	add	w0, w19, #123
	bl	srand
	sxtw	x20, w19
	sbfiz	x21, x19, #3, #32
	str	x23, [sp, #8]                   // 8-byte Folded Spill
	tbz	w19, #31, .LBB10_4
	b	.LBB10_9
.LBB10_3:
	mov	w19, #10000                     // =0x2710
	mov	w23, #300                       // =0x12c
	add	w0, w19, #123
	bl	srand
	sxtw	x20, w19
	sbfiz	x21, x19, #3, #32
	str	x23, [sp, #8]                   // 8-byte Folded Spill
	tbnz	w19, #31, .LBB10_9
.LBB10_4:
	mov	x0, x21
	bl	_Znam
	mov	x22, x0
	cbz	w20, .LBB10_10
// %bb.5:
	ubfiz	x19, x20, #3, #32
	mov	x23, xzr
.LBB10_6:                               // =>This Inner Loop Header: Depth=1
	bl	rand
	scvtf	d0, w0
	str	d0, [x22, x23]
	add	x23, x23, #8
	cmp	x19, x23
	b.ne	.LBB10_6
// %bb.7:
	ldr	x23, [sp, #8]                   // 8-byte Folded Reload
	mov	x0, x21
	bl	_Znam
	cmp	w23, #1
	mov	x27, x0
	b.ge	.LBB10_11
	b	.LBB10_191
.LBB10_8:
	ldr	x0, [x20, #16]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	mov	w19, w0
	add	w0, w19, #123
	bl	srand
	sxtw	x20, w19
	sbfiz	x21, x19, #3, #32
	str	x23, [sp, #8]                   // 8-byte Folded Spill
	tbz	w19, #31, .LBB10_4
.LBB10_9:
	mov	x0, #-1                         // =0xffffffffffffffff
	bl	_Znam
	mov	x22, x0
	mov	x0, #-1                         // =0xffffffffffffffff
	bl	_Znam
	cmp	w23, #1
	mov	x27, x0
	b.ge	.LBB10_11
	b	.LBB10_191
.LBB10_10:
	mov	x0, x21
	bl	_Znam
	cmp	w23, #1
	mov	x27, x0
	b.lt	.LBB10_191
.LBB10_11:
	mov	w26, wzr
	add	x28, x27, #8
	sub	x8, x21, #8
	adrp	x24, _Z19less_than_function1PKvS0_
	add	x24, x24, :lo12:_Z19less_than_function1PKvS0_
	adrp	x19, current_test
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	str	x8, [sp, #16]                   // 8-byte Folded Spill
	b	.LBB10_14
.LBB10_12:                              //   in Loop: Header=BB10_14 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_13:                              //   in Loop: Header=BB10_14 Depth=1
	add	w26, w26, #1
	cmp	w26, w23
	b.eq	.LBB10_21
.LBB10_14:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_17 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_19
// %bb.15:                              //   in Loop: Header=BB10_14 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_16:                              //   in Loop: Header=BB10_14 Depth=1
	mov	x0, x27
	mov	x1, x20
	mov	w2, #8                          // =0x8
	mov	x3, x24
	bl	qsort
	ldr	x8, [sp, #16]                   // 8-byte Folded Reload
	mov	x9, x28
.LBB10_17:                              //   Parent Loop BB10_14 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_13
// %bb.18:                              //   in Loop: Header=BB10_17 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_17
	b	.LBB10_12
.LBB10_19:                              //   in Loop: Header=BB10_14 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_16
// %bb.20:                              //   in Loop: Header=BB10_14 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_16
.LBB10_21:
	add	x8, x27, x20, lsl #3
	mov	w24, wzr
	adrp	x25, _Z19less_than_function2dd
	add	x25, x25, :lo12:_Z19less_than_function2dd
	add	x28, x27, #8
	adrp	x26, .L.str.11
	add	x26, x26, :lo12:.L.str.11
	stur	x8, [x29, #-8]                  // 8-byte Folded Spill
	b	.LBB10_24
.LBB10_22:                              //   in Loop: Header=BB10_24 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x26
	bl	printf
.LBB10_23:                              //   in Loop: Header=BB10_24 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_38
.LBB10_24:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_26 Depth 2
                                        //       Child Loop BB10_29 Depth 3
                                        //     Child Loop BB10_34 Depth 2
	cmp	w20, #1
	b.le	.LBB10_36
// %bb.25:                              //   in Loop: Header=BB10_24 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
	ldr	d0, [x27]
	ldur	x19, [x29, #-8]                 // 8-byte Folded Reload
	mov	x8, x27
.LBB10_26:                              //   Parent Loop BB10_24 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_29 Depth 3
	ldr	d1, [x19, #-8]!
	fcmp	d0, d1
	b.mi	.LBB10_26
// %bb.27:                              //   in Loop: Header=BB10_26 Depth=2
	cmp	x8, x19
	b.hs	.LBB10_32
// %bb.28:                              // %.preheader28
                                        //   in Loop: Header=BB10_26 Depth=2
	sub	x8, x8, #8
.LBB10_29:                              //   Parent Loop BB10_24 Depth=1
                                        //     Parent Loop BB10_26 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #8]!
	fcmp	d2, d0
	b.mi	.LBB10_29
// %bb.30:                              //   in Loop: Header=BB10_26 Depth=2
	cmp	x8, x19
	b.hs	.LBB10_32
// %bb.31:                              //   in Loop: Header=BB10_26 Depth=2
	str	d2, [x19]
	str	d1, [x8]
	b	.LBB10_26
.LBB10_32:                              //   in Loop: Header=BB10_24 Depth=1
	add	x1, x19, #8
	mov	x0, x27
	mov	x2, x25
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	add	x0, x19, #8
	mov	x2, x25
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	adrp	x19, current_test
.LBB10_33:                              // %.preheader29
                                        //   in Loop: Header=BB10_24 Depth=1
	ldr	x8, [sp, #16]                   // 8-byte Folded Reload
	mov	x9, x28
.LBB10_34:                              //   Parent Loop BB10_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_23
// %bb.35:                              //   in Loop: Header=BB10_34 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_34
	b	.LBB10_22
.LBB10_36:                              //   in Loop: Header=BB10_24 Depth=1
	b.ne	.LBB10_33
// %bb.37:                              //   in Loop: Header=BB10_24 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_33
.LBB10_38:
	ldr	x28, [sp, #16]                  // 8-byte Folded Reload
	mov	w24, wzr
	adrp	x25, _Z19less_than_function2dd
	add	x25, x25, :lo12:_Z19less_than_function2dd
	add	x19, x27, #8
	adrp	x26, .L.str.11
	add	x26, x26, :lo12:.L.str.11
	b	.LBB10_41
.LBB10_39:                              //   in Loop: Header=BB10_41 Depth=1
	adrp	x8, current_test
	mov	x0, x26
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB10_40:                              //   in Loop: Header=BB10_41 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_48
.LBB10_41:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_44 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_46
// %bb.42:                              //   in Loop: Header=BB10_41 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_43:                              //   in Loop: Header=BB10_41 Depth=1
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	mov	x0, x27
	mov	x2, x25
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	mov	x8, x28
	mov	x9, x19
.LBB10_44:                              //   Parent Loop BB10_41 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_40
// %bb.45:                              //   in Loop: Header=BB10_44 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_44
	b	.LBB10_39
.LBB10_46:                              //   in Loop: Header=BB10_41 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_43
// %bb.47:                              //   in Loop: Header=BB10_41 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_43
.LBB10_48:
	mov	w24, wzr
	adrp	x25, _Z19less_than_function2dd
	add	x25, x25, :lo12:_Z19less_than_function2dd
	add	x28, x27, #8
	adrp	x26, .L.str.11
	add	x26, x26, :lo12:.L.str.11
	adrp	x19, current_test
	b	.LBB10_51
.LBB10_49:                              //   in Loop: Header=BB10_51 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x26
	bl	printf
.LBB10_50:                              //   in Loop: Header=BB10_51 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_65
.LBB10_51:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_53 Depth 2
                                        //       Child Loop BB10_56 Depth 3
                                        //     Child Loop BB10_61 Depth 2
	cmp	w20, #1
	b.le	.LBB10_63
// %bb.52:                              //   in Loop: Header=BB10_51 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
	ldr	d0, [x27]
	ldur	x19, [x29, #-8]                 // 8-byte Folded Reload
	mov	x8, x27
.LBB10_53:                              //   Parent Loop BB10_51 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_56 Depth 3
	ldr	d1, [x19, #-8]!
	fcmp	d0, d1
	b.mi	.LBB10_53
// %bb.54:                              //   in Loop: Header=BB10_53 Depth=2
	cmp	x8, x19
	b.hs	.LBB10_59
// %bb.55:                              // %.preheader24
                                        //   in Loop: Header=BB10_53 Depth=2
	sub	x8, x8, #8
.LBB10_56:                              //   Parent Loop BB10_51 Depth=1
                                        //     Parent Loop BB10_53 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #8]!
	fcmp	d2, d0
	b.mi	.LBB10_56
// %bb.57:                              //   in Loop: Header=BB10_53 Depth=2
	cmp	x8, x19
	b.hs	.LBB10_59
// %bb.58:                              //   in Loop: Header=BB10_53 Depth=2
	str	d2, [x19]
	str	d1, [x8]
	b	.LBB10_53
.LBB10_59:                              //   in Loop: Header=BB10_51 Depth=1
	add	x1, x19, #8
	mov	x0, x27
	mov	x2, x25
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	add	x0, x19, #8
	mov	x2, x25
	bl	_Z9quicksortIPdPFbddEEvT_S3_T0_
	adrp	x19, current_test
.LBB10_60:                              // %.preheader25
                                        //   in Loop: Header=BB10_51 Depth=1
	ldr	x8, [sp, #16]                   // 8-byte Folded Reload
	mov	x9, x28
.LBB10_61:                              //   Parent Loop BB10_51 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_50
// %bb.62:                              //   in Loop: Header=BB10_61 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_61
	b	.LBB10_49
.LBB10_63:                              //   in Loop: Header=BB10_51 Depth=1
	b.ne	.LBB10_60
// %bb.64:                              //   in Loop: Header=BB10_51 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_60
.LBB10_65:
	clz	x8, x20
	ldr	x28, [sp, #16]                  // 8-byte Folded Reload
	mov	w24, wzr
	lsl	x8, x8, #1
	add	x25, x27, x21
	add	x19, x27, #8
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	add	x8, x27, #128
	str	x8, [sp]                        // 8-byte Folded Spill
	b	.LBB10_68
.LBB10_66:                              //   in Loop: Header=BB10_68 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.11
	add	x0, x0, :lo12:.L.str.11
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB10_67:                              //   in Loop: Header=BB10_68 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_103
.LBB10_68:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_97 Depth 2
                                        //       Child Loop BB10_100 Depth 3
                                        //     Child Loop BB10_74 Depth 2
                                        //       Child Loop BB10_79 Depth 3
                                        //     Child Loop BB10_82 Depth 2
                                        //       Child Loop BB10_84 Depth 3
                                        //     Child Loop BB10_89 Depth 2
	cmp	w20, #1
	b.le	.LBB10_85
// %bb.69:                              //   in Loop: Header=BB10_68 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	mov	x0, x27
	mov	x1, x25
	adrp	x3, _Z19less_than_function2dd
	add	x3, x3, :lo12:_Z19less_than_function2dd
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	cmp	w20, #17
	b.lt	.LBB10_92
// %bb.70:                              // %.preheader21
                                        //   in Loop: Header=BB10_68 Depth=1
	mov	x26, x19
	mov	w28, #8                         // =0x8
	mov	x23, x27
	b	.LBB10_74
.LBB10_71:                              //   in Loop: Header=BB10_74 Depth=2
	str	d0, [x8, #8]
.LBB10_72:                              //   in Loop: Header=BB10_74 Depth=2
	mov	x8, x27
.LBB10_73:                              //   in Loop: Header=BB10_74 Depth=2
	add	x28, x28, #8
	add	x26, x26, #8
	str	d8, [x8]
	cmp	x28, #128
	b.eq	.LBB10_80
.LBB10_74:                              //   Parent Loop BB10_68 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_79 Depth 3
	mov	x8, x23
	add	x23, x27, x28
	ldr	d0, [x27]
	ldr	d8, [x23]
	fcmp	d8, d0
	b.pl	.LBB10_77
// %bb.75:                              //   in Loop: Header=BB10_74 Depth=2
	cmp	x28, #9
	b.lo	.LBB10_71
// %bb.76:                              //   in Loop: Header=BB10_74 Depth=2
	mov	x0, x19
	mov	x1, x27
	mov	x2, x28
	bl	memmove
	b	.LBB10_72
.LBB10_77:                              //   in Loop: Header=BB10_74 Depth=2
	ldr	d0, [x8]
	mov	x8, x23
	fcmp	d8, d0
	b.pl	.LBB10_73
// %bb.78:                              // %.preheader14
                                        //   in Loop: Header=BB10_74 Depth=2
	mov	x8, x26
.LBB10_79:                              //   Parent Loop BB10_68 Depth=1
                                        //     Parent Loop BB10_74 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB10_79
	b	.LBB10_73
.LBB10_80:                              // %.preheader20
                                        //   in Loop: Header=BB10_68 Depth=1
	ldp	x8, x23, [sp]                   // 16-byte Folded Reload
	ldr	x28, [sp, #16]                  // 8-byte Folded Reload
	b	.LBB10_82
.LBB10_81:                              //   in Loop: Header=BB10_82 Depth=2
	add	x8, x8, #8
	str	d0, [x9]
	cmp	x8, x25
	b.eq	.LBB10_88
.LBB10_82:                              //   Parent Loop BB10_68 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_84 Depth 3
	ldp	d1, d0, [x8, #-8]
	mov	x9, x8
	fcmp	d0, d1
	b.pl	.LBB10_81
// %bb.83:                              // %.preheader12
                                        //   in Loop: Header=BB10_82 Depth=2
	mov	x9, x8
.LBB10_84:                              //   Parent Loop BB10_68 Depth=1
                                        //     Parent Loop BB10_82 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d1, [x9]
	ldur	d1, [x9, #-16]
	sub	x9, x9, #8
	fcmp	d0, d1
	b.mi	.LBB10_84
	b	.LBB10_81
.LBB10_85:                              //   in Loop: Header=BB10_68 Depth=1
	cbz	w20, .LBB10_88
// %bb.86:                              //   in Loop: Header=BB10_68 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_91
// %bb.87:                              //   in Loop: Header=BB10_68 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	ldr	d0, [x22]
	mov	x0, x27
	mov	x1, x25
	adrp	x3, _Z19less_than_function2dd
	add	x3, x3, :lo12:_Z19less_than_function2dd
	eor	x2, x8, #0x7e
	str	d0, [x27]
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
.LBB10_88:                              // %.preheader18
                                        //   in Loop: Header=BB10_68 Depth=1
	mov	x8, x28
	mov	x9, x19
.LBB10_89:                              //   Parent Loop BB10_68 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_67
// %bb.90:                              //   in Loop: Header=BB10_89 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_89
	b	.LBB10_66
.LBB10_91:                              //   in Loop: Header=BB10_68 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	mov	x0, x27
	mov	x1, x25
	adrp	x3, _Z19less_than_function2dd
	add	x3, x3, :lo12:_Z19less_than_function2dd
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
.LBB10_92:                              // %.preheader22
                                        //   in Loop: Header=BB10_68 Depth=1
	mov	x10, x19
	mov	x26, x27
	b	.LBB10_97
.LBB10_93:                              //   in Loop: Header=BB10_97 Depth=2
	sub	x2, x26, x27
	asr	x8, x2, #3
	cmp	x8, #2
	b.lt	.LBB10_101
// %bb.94:                              //   in Loop: Header=BB10_97 Depth=2
	sub	x8, x9, x8, lsl #3
	mov	x1, x27
	add	x0, x8, #16
	bl	memmove
.LBB10_95:                              //   in Loop: Header=BB10_97 Depth=2
	mov	x8, x27
.LBB10_96:                              //   in Loop: Header=BB10_97 Depth=2
	add	x10, x26, #8
	str	d8, [x8]
	cmp	x10, x25
	b.eq	.LBB10_88
.LBB10_97:                              //   Parent Loop BB10_68 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_100 Depth 3
	ldr	d8, [x10]
	ldr	d0, [x27]
	mov	x9, x26
	mov	x26, x10
	fcmp	d8, d0
	b.mi	.LBB10_93
// %bb.98:                              //   in Loop: Header=BB10_97 Depth=2
	ldr	d0, [x9]
	mov	x8, x26
	fcmp	d8, d0
	b.pl	.LBB10_96
// %bb.99:                              // %.preheader16
                                        //   in Loop: Header=BB10_97 Depth=2
	mov	x8, x26
.LBB10_100:                             //   Parent Loop BB10_68 Depth=1
                                        //     Parent Loop BB10_97 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB10_100
	b	.LBB10_96
.LBB10_101:                             //   in Loop: Header=BB10_97 Depth=2
	cmp	x2, #8
	mov	x8, x27
	b.ne	.LBB10_96
// %bb.102:                             //   in Loop: Header=BB10_97 Depth=2
	str	d0, [x9, #8]
	b	.LBB10_95
.LBB10_103:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	adrp	x19, current_test
	b	.LBB10_106
.LBB10_104:                             //   in Loop: Header=BB10_106 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_105:                             //   in Loop: Header=BB10_106 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_113
.LBB10_106:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_109 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_111
// %bb.107:                             //   in Loop: Header=BB10_106 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_108:                             //   in Loop: Header=BB10_106 Depth=1
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	mov	x0, x27
	bl	_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	mov	x8, x28
	mov	x9, x26
.LBB10_109:                             //   Parent Loop BB10_106 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_105
// %bb.110:                             //   in Loop: Header=BB10_109 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_109
	b	.LBB10_104
.LBB10_111:                             //   in Loop: Header=BB10_106 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_108
// %bb.112:                             //   in Loop: Header=BB10_106 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_108
.LBB10_113:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
.LBB10_114:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_117 Depth 2
	cbz	w20, .LBB10_116
// %bb.115:                             //   in Loop: Header=BB10_114 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	add	x1, x27, x21
	mov	x0, x27
	mov	x3, xzr
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	add	x1, x27, x21
	mov	x0, x27
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_
.LBB10_116:                             // %.preheader9
                                        //   in Loop: Header=BB10_114 Depth=1
	mov	x8, x28
	mov	x9, x26
.LBB10_117:                             //   Parent Loop BB10_114 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_120
// %bb.118:                             //   in Loop: Header=BB10_117 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_117
// %bb.119:                             //   in Loop: Header=BB10_114 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_120:                             //   in Loop: Header=BB10_114 Depth=1
	cmp	w20, #2
	b.lt	.LBB10_123
// %bb.121:                             //   in Loop: Header=BB10_114 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_122:                             //   in Loop: Header=BB10_114 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.ne	.LBB10_114
	b	.LBB10_125
.LBB10_123:                             //   in Loop: Header=BB10_114 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_122
// %bb.124:                             //   in Loop: Header=BB10_114 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_122
.LBB10_125:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	b	.LBB10_128
.LBB10_126:                             //   in Loop: Header=BB10_128 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_127:                             //   in Loop: Header=BB10_128 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_135
.LBB10_128:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_131 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_133
// %bb.129:                             //   in Loop: Header=BB10_128 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_130:                             //   in Loop: Header=BB10_128 Depth=1
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	mov	x0, x27
	bl	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	mov	x8, x28
	mov	x9, x26
.LBB10_131:                             //   Parent Loop BB10_128 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_127
// %bb.132:                             //   in Loop: Header=BB10_131 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_131
	b	.LBB10_126
.LBB10_133:                             //   in Loop: Header=BB10_128 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_130
// %bb.134:                             //   in Loop: Header=BB10_128 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_130
.LBB10_135:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	b	.LBB10_138
.LBB10_136:                             //   in Loop: Header=BB10_138 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_137:                             //   in Loop: Header=BB10_138 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_147
.LBB10_138:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_142 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_144
// %bb.139:                             //   in Loop: Header=BB10_138 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_140:                             //   in Loop: Header=BB10_138 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	add	x1, x27, x21
	mov	x0, x27
	mov	x3, xzr
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	add	x1, x27, x21
	mov	x0, x27
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_
.LBB10_141:                             // %.preheader6
                                        //   in Loop: Header=BB10_138 Depth=1
	mov	x8, x28
	mov	x9, x26
.LBB10_142:                             //   Parent Loop BB10_138 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_137
// %bb.143:                             //   in Loop: Header=BB10_142 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_142
	b	.LBB10_136
.LBB10_144:                             //   in Loop: Header=BB10_138 Depth=1
	cbz	w20, .LBB10_141
// %bb.145:                             //   in Loop: Header=BB10_138 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_140
// %bb.146:                             //   in Loop: Header=BB10_138 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_140
.LBB10_147:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	b	.LBB10_150
.LBB10_148:                             //   in Loop: Header=BB10_150 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_149:                             //   in Loop: Header=BB10_150 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_157
.LBB10_150:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_153 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_155
// %bb.151:                             //   in Loop: Header=BB10_150 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_152:                             //   in Loop: Header=BB10_150 Depth=1
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	mov	x0, x27
	bl	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	mov	x8, x28
	mov	x9, x26
.LBB10_153:                             //   Parent Loop BB10_150 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_149
// %bb.154:                             //   in Loop: Header=BB10_153 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_153
	b	.LBB10_148
.LBB10_155:                             //   in Loop: Header=BB10_150 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_152
// %bb.156:                             //   in Loop: Header=BB10_150 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_152
.LBB10_157:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	b	.LBB10_160
.LBB10_158:                             //   in Loop: Header=BB10_160 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_159:                             //   in Loop: Header=BB10_160 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_169
.LBB10_160:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_164 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_166
// %bb.161:                             //   in Loop: Header=BB10_160 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_162:                             //   in Loop: Header=BB10_160 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	add	x1, x27, x21
	mov	x0, x27
	mov	x3, xzr
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	add	x1, x27, x21
	mov	x0, x27
	mov	x2, xzr
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_
.LBB10_163:                             // %.preheader3
                                        //   in Loop: Header=BB10_160 Depth=1
	mov	x8, x28
	mov	x9, x26
.LBB10_164:                             //   Parent Loop BB10_160 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_159
// %bb.165:                             //   in Loop: Header=BB10_164 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_164
	b	.LBB10_158
.LBB10_166:                             //   in Loop: Header=BB10_160 Depth=1
	cbz	w20, .LBB10_163
// %bb.167:                             //   in Loop: Header=BB10_160 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_162
// %bb.168:                             //   in Loop: Header=BB10_160 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_162
.LBB10_169:
	mov	w24, wzr
	add	x26, x27, #8
	adrp	x25, .L.str.11
	add	x25, x25, :lo12:.L.str.11
	b	.LBB10_172
.LBB10_170:                             //   in Loop: Header=BB10_172 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x25
	bl	printf
.LBB10_171:                             //   in Loop: Header=BB10_172 Depth=1
	add	w24, w24, #1
	cmp	w24, w23
	b.eq	.LBB10_179
.LBB10_172:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_175 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_177
// %bb.173:                             //   in Loop: Header=BB10_172 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_174:                             //   in Loop: Header=BB10_172 Depth=1
	ldur	x1, [x29, #-8]                  // 8-byte Folded Reload
	mov	x0, x27
	bl	_Z9quicksortIPdEvT_S1_
	mov	x8, x28
	mov	x9, x26
.LBB10_175:                             //   Parent Loop BB10_172 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_171
// %bb.176:                             //   in Loop: Header=BB10_175 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_175
	b	.LBB10_170
.LBB10_177:                             //   in Loop: Header=BB10_172 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_174
// %bb.178:                             //   in Loop: Header=BB10_172 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_174
.LBB10_179:
	mov	w25, wzr
	add	x26, x27, #8
	adrp	x24, .L.str.11
	add	x24, x24, :lo12:.L.str.11
	b	.LBB10_182
.LBB10_180:                             //   in Loop: Header=BB10_182 Depth=1
	ldr	w1, [x19, :lo12:current_test]
	mov	x0, x24
	bl	printf
.LBB10_181:                             //   in Loop: Header=BB10_182 Depth=1
	add	w25, w25, #1
	cmp	w25, w23
	b.eq	.LBB10_191
.LBB10_182:                             // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_186 Depth 2
	cmp	w20, #2
	b.lt	.LBB10_188
// %bb.183:                             //   in Loop: Header=BB10_182 Depth=1
	mov	x0, x27
	mov	x1, x22
	mov	x2, x21
	bl	memcpy
.LBB10_184:                             //   in Loop: Header=BB10_182 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	add	x1, x27, x21
	mov	x0, x27
	eor	x2, x8, #0x7e
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	add	x1, x27, x21
	mov	x0, x27
	bl	_ZSt22__final_insertion_sortIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_
.LBB10_185:                             // %.preheader
                                        //   in Loop: Header=BB10_182 Depth=1
	mov	x8, x28
	mov	x9, x26
.LBB10_186:                             //   Parent Loop BB10_182 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB10_181
// %bb.187:                             //   in Loop: Header=BB10_186 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_186
	b	.LBB10_180
.LBB10_188:                             //   in Loop: Header=BB10_182 Depth=1
	cbz	w20, .LBB10_185
// %bb.189:                             //   in Loop: Header=BB10_182 Depth=1
	cmp	w20, #1
	b.ne	.LBB10_184
// %bb.190:                             //   in Loop: Header=BB10_182 Depth=1
	ldr	d0, [x22]
	str	d0, [x27]
	b	.LBB10_184
.LBB10_191:
	mov	x0, x27
	bl	_ZdaPv
	mov	x0, x22
	bl	_ZdaPv
	mov	w0, wzr
	.cfi_def_cfa wsp, 144
	ldp	x20, x19, [sp, #128]            // 16-byte Folded Reload
	ldr	d8, [sp, #32]                   // 8-byte Folded Reload
	ldp	x22, x21, [sp, #112]            // 16-byte Folded Reload
	ldp	x24, x23, [sp, #96]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #80]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #64]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #48]             // 16-byte Folded Reload
	add	sp, sp, #144
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
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB11_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x19, x1
	sub	x21, x1, #8
	b	.LBB11_3
.LBB11_2:                               //   in Loop: Header=BB11_3 Depth=1
	add	x20, x8, #8
	mov	x1, x20
	bl	_Z9quicksortIPd17less_than_functorEvT_S2_T0_
	cmp	x22, #8
	mov	x0, x20
	b.le	.LBB11_11
.LBB11_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_4 Depth 2
                                        //       Child Loop BB11_5 Depth 3
                                        //       Child Loop BB11_8 Depth 3
	ldr	d0, [x0]
	mov	x8, x19
	mov	x9, x0
.LBB11_4:                               //   Parent Loop BB11_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB11_5 Depth 3
                                        //       Child Loop BB11_8 Depth 3
	sub	x22, x21, x8
.LBB11_5:                               //   Parent Loop BB11_3 Depth=1
                                        //     Parent Loop BB11_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8, #-8]!
	add	x22, x22, #8
	fcmp	d0, d1
	b.mi	.LBB11_5
// %bb.6:                               //   in Loop: Header=BB11_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB11_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB11_4 Depth=2
	sub	x9, x9, #8
.LBB11_8:                               //   Parent Loop BB11_3 Depth=1
                                        //     Parent Loop BB11_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x9, #8]!
	fcmp	d2, d0
	b.mi	.LBB11_8
// %bb.9:                               //   in Loop: Header=BB11_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB11_2
// %bb.10:                              //   in Loop: Header=BB11_4 Depth=2
	str	d2, [x8]
	str	d1, [x9]
	b	.LBB11_4
.LBB11_11:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB12_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x19, x1
	sub	x21, x1, #8
	b	.LBB12_3
.LBB12_2:                               //   in Loop: Header=BB12_3 Depth=1
	add	x20, x8, #8
	mov	x1, x20
	bl	_Z9quicksortIPd24inline_less_than_functorEvT_S2_T0_
	cmp	x22, #8
	mov	x0, x20
	b.le	.LBB12_11
.LBB12_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB12_4 Depth 2
                                        //       Child Loop BB12_5 Depth 3
                                        //       Child Loop BB12_8 Depth 3
	ldr	d0, [x0]
	mov	x8, x19
	mov	x9, x0
.LBB12_4:                               //   Parent Loop BB12_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB12_5 Depth 3
                                        //       Child Loop BB12_8 Depth 3
	sub	x22, x21, x8
.LBB12_5:                               //   Parent Loop BB12_3 Depth=1
                                        //     Parent Loop BB12_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8, #-8]!
	add	x22, x22, #8
	fcmp	d0, d1
	b.mi	.LBB12_5
// %bb.6:                               //   in Loop: Header=BB12_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB12_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB12_4 Depth=2
	sub	x9, x9, #8
.LBB12_8:                               //   Parent Loop BB12_3 Depth=1
                                        //     Parent Loop BB12_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x9, #8]!
	fcmp	d2, d0
	b.mi	.LBB12_8
// %bb.9:                               //   in Loop: Header=BB12_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB12_2
// %bb.10:                              //   in Loop: Header=BB12_4 Depth=2
	str	d2, [x8]
	str	d1, [x9]
	b	.LBB12_4
.LBB12_11:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB13_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x19, x1
	sub	x21, x1, #8
	b	.LBB13_3
.LBB13_2:                               //   in Loop: Header=BB13_3 Depth=1
	add	x20, x8, #8
	mov	x1, x20
	bl	_Z9quicksortIPdSt4lessIdEEvT_S3_T0_
	cmp	x22, #8
	mov	x0, x20
	b.le	.LBB13_11
.LBB13_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_4 Depth 2
                                        //       Child Loop BB13_5 Depth 3
                                        //       Child Loop BB13_8 Depth 3
	ldr	d0, [x0]
	mov	x8, x19
	mov	x9, x0
.LBB13_4:                               //   Parent Loop BB13_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB13_5 Depth 3
                                        //       Child Loop BB13_8 Depth 3
	sub	x22, x21, x8
.LBB13_5:                               //   Parent Loop BB13_3 Depth=1
                                        //     Parent Loop BB13_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8, #-8]!
	add	x22, x22, #8
	fcmp	d0, d1
	b.mi	.LBB13_5
// %bb.6:                               //   in Loop: Header=BB13_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB13_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB13_4 Depth=2
	sub	x9, x9, #8
.LBB13_8:                               //   Parent Loop BB13_3 Depth=1
                                        //     Parent Loop BB13_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x9, #8]!
	fcmp	d2, d0
	b.mi	.LBB13_8
// %bb.9:                               //   in Loop: Header=BB13_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB13_2
// %bb.10:                              //   in Loop: Header=BB13_4 Depth=2
	str	d2, [x8]
	str	d1, [x9]
	b	.LBB13_4
.LBB13_11:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
	sub	x8, x1, x0
	cmp	x8, #9
	b.lt	.LBB14_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x19, x1
	sub	x21, x1, #8
	b	.LBB14_3
.LBB14_2:                               //   in Loop: Header=BB14_3 Depth=1
	add	x20, x8, #8
	mov	x1, x20
	bl	_Z9quicksortIPdEvT_S1_
	cmp	x22, #8
	mov	x0, x20
	b.le	.LBB14_11
.LBB14_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_4 Depth 2
                                        //       Child Loop BB14_5 Depth 3
                                        //       Child Loop BB14_8 Depth 3
	ldr	d0, [x0]
	mov	x8, x19
	mov	x9, x0
.LBB14_4:                               //   Parent Loop BB14_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB14_5 Depth 3
                                        //       Child Loop BB14_8 Depth 3
	sub	x22, x21, x8
.LBB14_5:                               //   Parent Loop BB14_3 Depth=1
                                        //     Parent Loop BB14_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x8, #-8]!
	add	x22, x22, #8
	fcmp	d0, d1
	b.mi	.LBB14_5
// %bb.6:                               //   in Loop: Header=BB14_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB14_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB14_4 Depth=2
	sub	x9, x9, #8
.LBB14_8:                               //   Parent Loop BB14_3 Depth=1
                                        //     Parent Loop BB14_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x9, #8]!
	fcmp	d2, d0
	b.mi	.LBB14_8
// %bb.9:                               //   in Loop: Header=BB14_4 Depth=2
	cmp	x9, x8
	b.hs	.LBB14_2
// %bb.10:                              //   in Loop: Header=BB14_4 Depth=2
	str	d2, [x8]
	str	d1, [x9]
	b	.LBB14_4
.LBB14_11:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
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
	str	d8, [sp, #-112]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 112
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #32]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             // 16-byte Folded Spill
	add	x29, sp, #16
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
	sub	x26, x1, x0
	cmp	x26, #129
	b.lt	.LBB15_34
// %bb.1:
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x19, x3
	mov	x21, x1
	mov	x20, x0
	mov	x22, x2
	add	x24, x0, #8
	sub	x25, x8, x0
	b	.LBB15_3
.LBB15_2:                               //   in Loop: Header=BB15_3 Depth=1
	mov	x0, x23
	mov	x1, x21
	mov	x2, x22
	mov	x3, x19
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_T0_T1_
	cmp	x26, #128
	mov	x21, x23
	b.le	.LBB15_34
.LBB15_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_15 Depth 2
                                        //       Child Loop BB15_16 Depth 3
                                        //       Child Loop BB15_18 Depth 3
	cbz	x22, .LBB15_21
// %bb.4:                               //   in Loop: Header=BB15_3 Depth=1
	lsr	x23, x26, #4
	ldr	d0, [x20, #8]
	ldr	d1, [x20, x23, lsl #3]
	blr	x19
	ldur	d1, [x21, #-8]
	tbz	w0, #0, .LBB15_7
// %bb.5:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x20, x23, lsl #3]
	blr	x19
	tbz	w0, #0, .LBB15_9
// %bb.6:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x20, x23, lsl #3]
	ldr	d1, [x20]
	str	d0, [x20]
	str	d1, [x20, x23, lsl #3]
	b	.LBB15_14
.LBB15_7:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x24]
	blr	x19
	tbz	w0, #0, .LBB15_11
// %bb.8:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	q0, [x20]
	ext	v0.16b, v0.16b, v0.16b, #8
	str	q0, [x20]
	b	.LBB15_14
.LBB15_9:                               //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x20, #8]
	ldur	d1, [x21, #-8]
	blr	x19
	ldr	d0, [x20]
	tbnz	w0, #0, .LBB15_12
// %bb.10:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x20, #8]
	stp	d1, d0, [x20]
	b	.LBB15_14
.LBB15_11:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d0, [x20, x23, lsl #3]
	ldur	d1, [x21, #-8]
	blr	x19
	ldr	d0, [x20]
	tbz	w0, #0, .LBB15_13
.LBB15_12:                              //   in Loop: Header=BB15_3 Depth=1
	ldur	d1, [x21, #-8]
	str	d1, [x20]
	stur	d0, [x21, #-8]
	b	.LBB15_14
.LBB15_13:                              //   in Loop: Header=BB15_3 Depth=1
	ldr	d1, [x20, x23, lsl #3]
	str	d1, [x20]
	str	d0, [x20, x23, lsl #3]
.LBB15_14:                              // %.preheader6
                                        //   in Loop: Header=BB15_3 Depth=1
	sub	x22, x22, #1
	mov	x27, x21
	mov	x28, x24
.LBB15_15:                              //   Parent Loop BB15_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB15_16 Depth 3
                                        //       Child Loop BB15_18 Depth 3
	add	x26, x25, x28
	sub	x23, x28, #8
.LBB15_16:                              //   Parent Loop BB15_3 Depth=1
                                        //     Parent Loop BB15_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d0, [x23, #8]!
	ldr	d1, [x20]
	blr	x19
	add	x26, x26, #8
	tbnz	w0, #0, .LBB15_16
// %bb.17:                              // %.preheader5
                                        //   in Loop: Header=BB15_15 Depth=2
	add	x28, x23, #8
.LBB15_18:                              //   Parent Loop BB15_3 Depth=1
                                        //     Parent Loop BB15_15 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x27, #-8]!
	ldr	d0, [x20]
	blr	x19
	tbnz	w0, #0, .LBB15_18
// %bb.19:                              //   in Loop: Header=BB15_15 Depth=2
	cmp	x23, x27
	b.hs	.LBB15_2
// %bb.20:                              //   in Loop: Header=BB15_15 Depth=2
	ldr	d0, [x27]
	ldr	d1, [x23]
	str	d0, [x23]
	str	d1, [x27]
	b	.LBB15_15
.LBB15_21:
	add	x2, sp, #8
	mov	x0, x20
	mov	x1, x21
	str	x19, [sp, #8]
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterIPFbddEEEEvT_S7_RT0_
	mov	w22, #1                         // =0x1
	b	.LBB15_24
.LBB15_22:                              //   in Loop: Header=BB15_24 Depth=1
	mov	x24, xzr
.LBB15_23:                              //   in Loop: Header=BB15_24 Depth=1
	cmp	x23, #8
	str	d8, [x20, x24, lsl #3]
	b.le	.LBB15_34
.LBB15_24:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_26 Depth 2
                                        //     Child Loop BB15_32 Depth 2
	ldr	d8, [x21, #-8]!
	sub	x23, x21, x20
	asr	x25, x23, #3
	ldr	d0, [x20]
	cmp	x25, #3
	str	d0, [x21]
	b.lt	.LBB15_28
// %bb.25:                              // %.preheader2
                                        //   in Loop: Header=BB15_24 Depth=1
	sub	x8, x25, #1
	mov	x27, xzr
	add	x8, x8, x8, lsr #63
	asr	x26, x8, #1
.LBB15_26:                              //   Parent Loop BB15_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x24, x27, #1
	add	x8, x20, x27, lsl #4
	add	x28, x24, #2
	ldr	d1, [x8, #8]
	ldr	d0, [x20, x28, lsl #3]
	blr	x19
	orr	x8, x24, #0x1
	tst	w0, #0x1
	csel	x24, x28, x8, eq
	ldr	d0, [x20, x24, lsl #3]
	cmp	x24, x26
	str	d0, [x20, x27, lsl #3]
	mov	x27, x24
	b.lt	.LBB15_26
// %bb.27:                              //   in Loop: Header=BB15_24 Depth=1
	tbz	w23, #3, .LBB15_29
	b	.LBB15_31
.LBB15_28:                              //   in Loop: Header=BB15_24 Depth=1
	mov	x24, xzr
	tbnz	w23, #3, .LBB15_31
.LBB15_29:                              //   in Loop: Header=BB15_24 Depth=1
	sub	x8, x25, #2
	cmp	x24, x8, asr #1
	b.ne	.LBB15_31
// %bb.30:                              //   in Loop: Header=BB15_24 Depth=1
	orr	x8, x22, x24, lsl #1
	ldr	d0, [x20, x8, lsl #3]
	str	d0, [x20, x24, lsl #3]
	mov	x24, x8
	b	.LBB15_32
.LBB15_31:                              //   in Loop: Header=BB15_24 Depth=1
	cbz	x24, .LBB15_23
.LBB15_32:                              //   Parent Loop BB15_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x26, x24, #1
	fmov	d1, d8
	lsr	x25, x26, #1
	ldr	d0, [x20, x25, lsl #3]
	blr	x19
	tbz	w0, #0, .LBB15_23
// %bb.33:                              //   in Loop: Header=BB15_32 Depth=2
	ldr	d0, [x20, x25, lsl #3]
	cmp	x26, #1
	str	d0, [x20, x24, lsl #3]
	mov	x24, x25
	b.hi	.LBB15_32
	b	.LBB15_22
.LBB15_34:
	.cfi_def_cfa wsp, 112
	ldp	x20, x19, [sp, #96]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #112                  // 8-byte Folded Reload
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
	sub	x9, x1, x0
	asr	x10, x9, #3
	subs	x8, x10, #2
	b.lt	.LBB16_22
// %bb.1:
	str	d8, [sp, #-112]!                // 8-byte Folded Spill
	.cfi_def_cfa_offset 112
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x28, x27, [sp, #32]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #48]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #64]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #80]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #96]             // 16-byte Folded Spill
	add	x29, sp, #16
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
	sub	x10, x10, #1
	lsr	x22, x8, #1
	mov	x19, x0
	lsr	x21, x10, #1
	mov	x20, x2
	tbnz	w9, #3, .LBB16_15
// %bb.2:
	orr	x23, x8, #0x1
	mov	x8, x22
	b	.LBB16_5
.LBB16_3:                               //   in Loop: Header=BB16_5 Depth=1
	mov	x27, x26
.LBB16_4:                               //   in Loop: Header=BB16_5 Depth=1
	sub	x8, x24, #1
	str	d8, [x19, x27, lsl #3]
	cbz	x24, .LBB16_21
.LBB16_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_7 Depth 2
                                        //     Child Loop BB16_11 Depth 2
	ldr	d8, [x19, x8, lsl #3]
	ldr	x25, [x20]
	mov	x24, x8
	cmp	x8, x21
	mov	x26, x8
	b.ge	.LBB16_8
// %bb.6:                               // %.preheader1
                                        //   in Loop: Header=BB16_5 Depth=1
	mov	x27, x24
.LBB16_7:                               //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x26, x27, #1
	add	x8, x19, x27, lsl #4
	add	x28, x26, #2
	ldr	d1, [x8, #8]
	ldr	d0, [x19, x28, lsl #3]
	blr	x25
	orr	x8, x26, #0x1
	tst	w0, #0x1
	csel	x26, x28, x8, eq
	ldr	d0, [x19, x26, lsl #3]
	cmp	x26, x21
	str	d0, [x19, x27, lsl #3]
	mov	x27, x26
	b.lt	.LBB16_7
.LBB16_8:                               //   in Loop: Header=BB16_5 Depth=1
	cmp	x26, x22
	b.ne	.LBB16_10
// %bb.9:                               //   in Loop: Header=BB16_5 Depth=1
	ldr	d0, [x19, x23, lsl #3]
	mov	x26, x23
	str	d0, [x19, x22, lsl #3]
.LBB16_10:                              //   in Loop: Header=BB16_5 Depth=1
	cmp	x26, x24
	b.le	.LBB16_3
.LBB16_11:                              //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x8, x26, #1
	fmov	d1, d8
	add	x8, x8, x8, lsr #63
	asr	x27, x8, #1
	ldr	d0, [x19, x27, lsl #3]
	blr	x25
	tbz	w0, #0, .LBB16_3
// %bb.12:                              //   in Loop: Header=BB16_11 Depth=2
	ldr	d0, [x19, x27, lsl #3]
	cmp	x27, x24
	str	d0, [x19, x26, lsl #3]
	mov	x26, x27
	b.gt	.LBB16_11
	b	.LBB16_4
.LBB16_13:                              //   in Loop: Header=BB16_15 Depth=1
	mov	x25, x24
.LBB16_14:                              //   in Loop: Header=BB16_15 Depth=1
	sub	x22, x23, #1
	str	d8, [x19, x25, lsl #3]
	cbz	x23, .LBB16_21
.LBB16_15:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_17 Depth 2
                                        //     Child Loop BB16_19 Depth 2
	ldr	d8, [x19, x22, lsl #3]
	mov	x23, x22
	cmp	x22, x21
	mov	x25, x22
	b.ge	.LBB16_14
// %bb.16:                              // %.preheader7
                                        //   in Loop: Header=BB16_15 Depth=1
	ldr	x22, [x20]
	mov	x24, x23
.LBB16_17:                              //   Parent Loop BB16_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	mov	x25, x24
	lsl	x24, x24, #1
	add	x8, x19, x25, lsl #4
	add	x26, x24, #2
	ldr	d0, [x19, x26, lsl #3]
	ldr	d1, [x8, #8]
	blr	x22
	orr	x8, x24, #0x1
	tst	w0, #0x1
	csel	x24, x26, x8, eq
	ldr	d0, [x19, x24, lsl #3]
	cmp	x24, x21
	str	d0, [x19, x25, lsl #3]
	b.lt	.LBB16_17
// %bb.18:                              //   in Loop: Header=BB16_15 Depth=1
	cmp	x24, x23
	b.le	.LBB16_13
.LBB16_19:                              //   Parent Loop BB16_15 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x8, x24, #1
	fmov	d1, d8
	add	x8, x8, x8, lsr #63
	asr	x25, x8, #1
	ldr	d0, [x19, x25, lsl #3]
	blr	x22
	tbz	w0, #0, .LBB16_13
// %bb.20:                              //   in Loop: Header=BB16_19 Depth=2
	ldr	d0, [x19, x25, lsl #3]
	cmp	x25, x23
	str	d0, [x19, x24, lsl #3]
	mov	x24, x25
	b.gt	.LBB16_19
	b	.LBB16_14
.LBB16_21:
	.cfi_def_cfa wsp, 112
	ldp	x20, x19, [sp, #96]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #80]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #64]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #48]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #112                  // 8-byte Folded Reload
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
	stp	x29, x30, [sp, #-80]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	str	x25, [sp, #16]                  // 8-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 80
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -64
	.cfi_offset w30, -72
	.cfi_offset w29, -80
	sub	x25, x1, x0
	cmp	x25, #129
	b.lt	.LBB17_37
// %bb.1:
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x20, x3
	mov	x19, x0
	mov	x21, x2
	add	x23, x0, #8
	sub	x24, x8, x0
	b	.LBB17_3
.LBB17_2:                               //   in Loop: Header=BB17_3 Depth=1
	and	x3, x20, #0xff
	mov	x0, x22
	mov	x2, x21
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_T0_T1_
	cmp	x25, #128
	mov	x1, x22
	b.le	.LBB17_37
.LBB17_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_16 Depth 2
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_19 Depth 3
	cbz	x21, .LBB17_22
// %bb.4:                               //   in Loop: Header=BB17_3 Depth=1
	lsr	x8, x25, #4
	ldr	d1, [x19, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x19, x8, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB17_7
// %bb.5:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB17_9
// %bb.6:                               //   in Loop: Header=BB17_3 Depth=1
	ldr	d0, [x19]
	str	d2, [x19]
	str	d0, [x19, x8, lsl #3]
	b	.LBB17_15
.LBB17_7:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB17_11
// %bb.8:                               //   in Loop: Header=BB17_3 Depth=1
	ldr	d0, [x19]
	stp	d1, d0, [x19]
	b	.LBB17_15
.LBB17_9:                               //   in Loop: Header=BB17_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x19]
	b.pl	.LBB17_13
// %bb.10:                              //   in Loop: Header=BB17_3 Depth=1
	str	d0, [x19]
	stur	d2, [x1, #-8]
	b	.LBB17_15
.LBB17_11:                              //   in Loop: Header=BB17_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x19]
	b.pl	.LBB17_14
// %bb.12:                              //   in Loop: Header=BB17_3 Depth=1
	str	d0, [x19]
	stur	d1, [x1, #-8]
	b	.LBB17_15
.LBB17_13:                              //   in Loop: Header=BB17_3 Depth=1
	stp	d1, d2, [x19]
	b	.LBB17_15
.LBB17_14:                              //   in Loop: Header=BB17_3 Depth=1
	str	d2, [x19]
	str	d1, [x19, x8, lsl #3]
.LBB17_15:                              // %.preheader6
                                        //   in Loop: Header=BB17_3 Depth=1
	sub	x21, x21, #1
	mov	x8, x1
	mov	x9, x23
.LBB17_16:                              //   Parent Loop BB17_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB17_17 Depth 3
                                        //       Child Loop BB17_19 Depth 3
	ldr	d0, [x19]
	add	x25, x24, x9
.LBB17_17:                              //   Parent Loop BB17_3 Depth=1
                                        //     Parent Loop BB17_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x9], #8
	add	x25, x25, #8
	fcmp	d1, d0
	b.mi	.LBB17_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB17_16 Depth=2
	sub	x22, x9, #8
.LBB17_19:                              //   Parent Loop BB17_3 Depth=1
                                        //     Parent Loop BB17_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #-8]!
	fcmp	d0, d2
	b.mi	.LBB17_19
// %bb.20:                              //   in Loop: Header=BB17_16 Depth=2
	cmp	x22, x8
	b.hs	.LBB17_2
// %bb.21:                              //   in Loop: Header=BB17_16 Depth=2
	str	d2, [x22]
	str	d1, [x8]
	b	.LBB17_16
.LBB17_22:
	add	x2, x29, #28
	mov	x0, x19
	strb	w20, [x29, #28]
	mov	x20, x1
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI17less_than_functorEEEvT_S6_RT0_
	mov	w8, #1                          // =0x1
	b	.LBB17_25
.LBB17_23:                              //   in Loop: Header=BB17_25 Depth=1
	mov	x10, xzr
.LBB17_24:                              //   in Loop: Header=BB17_25 Depth=1
	cmp	x9, #8
	str	d0, [x19, x10, lsl #3]
	b.le	.LBB17_37
.LBB17_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_28 Depth 2
                                        //     Child Loop BB17_35 Depth 2
	ldr	d0, [x20, #-8]!
	sub	x9, x20, x19
	asr	x11, x9, #3
	ldr	d1, [x19]
	cmp	x11, #3
	str	d1, [x20]
	b.lt	.LBB17_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB17_25 Depth=1
	sub	x10, x11, #1
	mov	x13, xzr
	add	x10, x10, x10, lsr #63
	asr	x12, x10, #1
	b	.LBB17_28
.LBB17_27:                              // %select.end
                                        //   in Loop: Header=BB17_28 Depth=2
	ldr	d1, [x19, x10, lsl #3]
	cmp	x10, x12
	str	d1, [x19, x13, lsl #3]
	mov	x13, x10
	b.ge	.LBB17_31
.LBB17_28:                              //   Parent Loop BB17_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x19, x13, lsl #4
	add	x10, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x19, x10, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB17_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB17_28 Depth=2
	orr	x10, x14, #0x1
	b	.LBB17_27
.LBB17_30:                              //   in Loop: Header=BB17_25 Depth=1
	mov	x10, xzr
.LBB17_31:                              //   in Loop: Header=BB17_25 Depth=1
	tbnz	w9, #3, .LBB17_34
// %bb.32:                              //   in Loop: Header=BB17_25 Depth=1
	sub	x11, x11, #2
	cmp	x10, x11, asr #1
	b.ne	.LBB17_34
// %bb.33:                              //   in Loop: Header=BB17_25 Depth=1
	orr	x11, x8, x10, lsl #1
	ldr	d1, [x19, x11, lsl #3]
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b	.LBB17_35
.LBB17_34:                              //   in Loop: Header=BB17_25 Depth=1
	cbz	x10, .LBB17_24
.LBB17_35:                              //   Parent Loop BB17_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x19, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB17_24
// %bb.36:                              //   in Loop: Header=BB17_35 Depth=2
	cmp	x12, #1
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB17_35
	b	.LBB17_23
.LBB17_37:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldr	x25, [sp, #16]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #80             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w30
	.cfi_restore w29
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
	str	d8, [sp, #-80]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -80
	.cfi_remember_state
	sub	x8, x1, x0
	mov	x19, x1
	mov	x20, x0
	cmp	x8, #129
	b.lt	.LBB18_2
// %bb.1:
	add	x21, x20, #8
	mov	w22, #8                         // =0x8
	mov	x24, x20
	mov	x23, x21
	b	.LBB18_18
.LBB18_2:
	cmp	x20, x19
	b.eq	.LBB18_25
// %bb.3:
	add	x10, x20, #8
	cmp	x10, x19
	b.eq	.LBB18_25
// %bb.4:                               // %.preheader7
	mov	x21, x20
	b	.LBB18_9
.LBB18_5:                               //   in Loop: Header=BB18_9 Depth=1
	sub	x2, x21, x20
	asr	x8, x2, #3
	cmp	x8, #2
	b.lt	.LBB18_13
// %bb.6:                               //   in Loop: Header=BB18_9 Depth=1
	sub	x8, x9, x8, lsl #3
	mov	x1, x20
	add	x0, x8, #16
	bl	memmove
.LBB18_7:                               //   in Loop: Header=BB18_9 Depth=1
	mov	x8, x20
.LBB18_8:                               //   in Loop: Header=BB18_9 Depth=1
	add	x10, x21, #8
	str	d8, [x8]
	cmp	x10, x19
	b.eq	.LBB18_25
.LBB18_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_12 Depth 2
	ldr	d8, [x21, #8]
	ldr	d0, [x20]
	mov	x9, x21
	mov	x21, x10
	fcmp	d8, d0
	b.mi	.LBB18_5
// %bb.10:                              //   in Loop: Header=BB18_9 Depth=1
	ldr	d0, [x9]
	mov	x8, x21
	fcmp	d8, d0
	b.pl	.LBB18_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB18_9 Depth=1
	mov	x8, x21
.LBB18_12:                              //   Parent Loop BB18_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB18_12
	b	.LBB18_8
.LBB18_13:                              //   in Loop: Header=BB18_9 Depth=1
	cmp	x2, #8
	mov	x8, x20
	b.ne	.LBB18_8
// %bb.14:                              //   in Loop: Header=BB18_9 Depth=1
	str	d0, [x9, #8]
	b	.LBB18_7
.LBB18_15:                              //   in Loop: Header=BB18_18 Depth=1
	str	d0, [x8, #8]
.LBB18_16:                              //   in Loop: Header=BB18_18 Depth=1
	mov	x8, x20
.LBB18_17:                              //   in Loop: Header=BB18_18 Depth=1
	add	x22, x22, #8
	add	x23, x23, #8
	str	d8, [x8]
	cmp	x22, #128
	b.eq	.LBB18_24
.LBB18_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_23 Depth 2
	mov	x8, x24
	add	x24, x20, x22
	ldr	d0, [x20]
	ldr	d8, [x24]
	fcmp	d8, d0
	b.pl	.LBB18_21
// %bb.19:                              //   in Loop: Header=BB18_18 Depth=1
	cmp	x22, #9
	b.lo	.LBB18_15
// %bb.20:                              //   in Loop: Header=BB18_18 Depth=1
	mov	x0, x21
	mov	x1, x20
	mov	x2, x22
	bl	memmove
	b	.LBB18_16
.LBB18_21:                              //   in Loop: Header=BB18_18 Depth=1
	ldr	d0, [x8]
	mov	x8, x24
	fcmp	d8, d0
	b.pl	.LBB18_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB18_18 Depth=1
	mov	x8, x23
.LBB18_23:                              //   Parent Loop BB18_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB18_23
	b	.LBB18_17
.LBB18_24:
	add	x8, x20, #128
	cmp	x8, x19
	b.ne	.LBB18_27
.LBB18_25:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #80                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB18_26:                              //   in Loop: Header=BB18_27 Depth=1
	.cfi_restore_state
	add	x8, x8, #8
	str	d0, [x9]
	cmp	x8, x19
	b.eq	.LBB18_25
.LBB18_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_29 Depth 2
	ldp	d1, d0, [x8, #-8]
	mov	x9, x8
	fcmp	d0, d1
	b.pl	.LBB18_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB18_27 Depth=1
	mov	x9, x8
.LBB18_29:                              //   Parent Loop BB18_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x9]
	ldur	d1, [x9, #-16]
	sub	x9, x9, #8
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
	sub	x11, x1, x0
	asr	x8, x11, #3
	subs	x10, x8, #2
	b.ge	.LBB19_2
.LBB19_1:
	ret
.LBB19_2:
	sub	x8, x8, #1
	lsr	x9, x10, #1
	lsr	x8, x8, #1
	tbnz	w11, #3, .LBB19_18
// %bb.3:
	orr	x10, x10, #0x1
	mov	x12, x9
	b	.LBB19_6
.LBB19_4:                               //   in Loop: Header=BB19_6 Depth=1
	mov	x13, x12
.LBB19_5:                               //   in Loop: Header=BB19_6 Depth=1
	sub	x12, x11, #1
	str	d0, [x0, x13, lsl #3]
	cbz	x11, .LBB19_1
.LBB19_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_9 Depth 2
                                        //     Child Loop BB19_14 Depth 2
	ldr	d0, [x0, x12, lsl #3]
	mov	x11, x12
	cmp	x12, x8
	b.ge	.LBB19_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB19_6 Depth=1
	mov	x13, x11
	b	.LBB19_9
.LBB19_8:                               // %select.end
                                        //   in Loop: Header=BB19_9 Depth=2
	ldr	d1, [x0, x12, lsl #3]
	cmp	x12, x8
	str	d1, [x0, x13, lsl #3]
	mov	x13, x12
	b.ge	.LBB19_11
.LBB19_9:                               //   Parent Loop BB19_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x0, x13, lsl #4
	add	x12, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x0, x12, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB19_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB19_9 Depth=2
	orr	x12, x14, #0x1
	b	.LBB19_8
.LBB19_11:                              //   in Loop: Header=BB19_6 Depth=1
	cmp	x12, x9
	b.ne	.LBB19_13
// %bb.12:                              //   in Loop: Header=BB19_6 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	mov	x12, x10
	str	d1, [x0, x9, lsl #3]
.LBB19_13:                              //   in Loop: Header=BB19_6 Depth=1
	cmp	x12, x11
	b.le	.LBB19_4
.LBB19_14:                              //   Parent Loop BB19_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_4
// %bb.15:                              //   in Loop: Header=BB19_14 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.gt	.LBB19_14
	b	.LBB19_5
.LBB19_16:                              //   in Loop: Header=BB19_18 Depth=1
	mov	x11, x9
.LBB19_17:                              //   in Loop: Header=BB19_18 Depth=1
	sub	x9, x10, #1
	str	d0, [x0, x11, lsl #3]
	cbz	x10, .LBB19_1
.LBB19_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_21 Depth 2
                                        //     Child Loop BB19_24 Depth 2
	ldr	d0, [x0, x9, lsl #3]
	mov	x10, x9
	cmp	x9, x8
	mov	x11, x9
	b.ge	.LBB19_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB19_18 Depth=1
	mov	x11, x10
	b	.LBB19_21
.LBB19_20:                              // %select.end10
                                        //   in Loop: Header=BB19_21 Depth=2
	ldr	d1, [x0, x9, lsl #3]
	cmp	x9, x8
	str	d1, [x0, x11, lsl #3]
	mov	x11, x9
	b.ge	.LBB19_23
.LBB19_21:                              //   Parent Loop BB19_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x12, x11, #1
	add	x13, x0, x11, lsl #4
	add	x9, x12, #2
	ldr	d2, [x13, #8]
	ldr	d1, [x0, x9, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB19_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB19_21 Depth=2
	orr	x9, x12, #0x1
	b	.LBB19_20
.LBB19_23:                              //   in Loop: Header=BB19_18 Depth=1
	cmp	x9, x10
	b.le	.LBB19_16
.LBB19_24:                              //   Parent Loop BB19_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x11, x9, #1
	add	x11, x11, x11, lsr #63
	asr	x11, x11, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_16
// %bb.25:                              //   in Loop: Header=BB19_24 Depth=2
	cmp	x11, x10
	str	d1, [x0, x9, lsl #3]
	mov	x9, x11
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
	stp	x29, x30, [sp, #-80]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	str	x25, [sp, #16]                  // 8-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 80
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -64
	.cfi_offset w30, -72
	.cfi_offset w29, -80
	sub	x25, x1, x0
	cmp	x25, #129
	b.lt	.LBB20_37
// %bb.1:
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x20, x3
	mov	x19, x0
	mov	x21, x2
	add	x23, x0, #8
	sub	x24, x8, x0
	b	.LBB20_3
.LBB20_2:                               //   in Loop: Header=BB20_3 Depth=1
	and	x3, x20, #0xff
	mov	x0, x22
	mov	x2, x21
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_T0_T1_
	cmp	x25, #128
	mov	x1, x22
	b.le	.LBB20_37
.LBB20_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_16 Depth 2
                                        //       Child Loop BB20_17 Depth 3
                                        //       Child Loop BB20_19 Depth 3
	cbz	x21, .LBB20_22
// %bb.4:                               //   in Loop: Header=BB20_3 Depth=1
	lsr	x8, x25, #4
	ldr	d1, [x19, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x19, x8, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB20_7
// %bb.5:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB20_9
// %bb.6:                               //   in Loop: Header=BB20_3 Depth=1
	ldr	d0, [x19]
	str	d2, [x19]
	str	d0, [x19, x8, lsl #3]
	b	.LBB20_15
.LBB20_7:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB20_11
// %bb.8:                               //   in Loop: Header=BB20_3 Depth=1
	ldr	d0, [x19]
	stp	d1, d0, [x19]
	b	.LBB20_15
.LBB20_9:                               //   in Loop: Header=BB20_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x19]
	b.pl	.LBB20_13
// %bb.10:                              //   in Loop: Header=BB20_3 Depth=1
	str	d0, [x19]
	stur	d2, [x1, #-8]
	b	.LBB20_15
.LBB20_11:                              //   in Loop: Header=BB20_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x19]
	b.pl	.LBB20_14
// %bb.12:                              //   in Loop: Header=BB20_3 Depth=1
	str	d0, [x19]
	stur	d1, [x1, #-8]
	b	.LBB20_15
.LBB20_13:                              //   in Loop: Header=BB20_3 Depth=1
	stp	d1, d2, [x19]
	b	.LBB20_15
.LBB20_14:                              //   in Loop: Header=BB20_3 Depth=1
	str	d2, [x19]
	str	d1, [x19, x8, lsl #3]
.LBB20_15:                              // %.preheader6
                                        //   in Loop: Header=BB20_3 Depth=1
	sub	x21, x21, #1
	mov	x8, x1
	mov	x9, x23
.LBB20_16:                              //   Parent Loop BB20_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB20_17 Depth 3
                                        //       Child Loop BB20_19 Depth 3
	ldr	d0, [x19]
	add	x25, x24, x9
.LBB20_17:                              //   Parent Loop BB20_3 Depth=1
                                        //     Parent Loop BB20_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x9], #8
	add	x25, x25, #8
	fcmp	d1, d0
	b.mi	.LBB20_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB20_16 Depth=2
	sub	x22, x9, #8
.LBB20_19:                              //   Parent Loop BB20_3 Depth=1
                                        //     Parent Loop BB20_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #-8]!
	fcmp	d0, d2
	b.mi	.LBB20_19
// %bb.20:                              //   in Loop: Header=BB20_16 Depth=2
	cmp	x22, x8
	b.hs	.LBB20_2
// %bb.21:                              //   in Loop: Header=BB20_16 Depth=2
	str	d2, [x22]
	str	d1, [x8]
	b	.LBB20_16
.LBB20_22:
	add	x2, x29, #28
	mov	x0, x19
	strb	w20, [x29, #28]
	mov	x20, x1
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterI24inline_less_than_functorEEEvT_S6_RT0_
	mov	w8, #1                          // =0x1
	b	.LBB20_25
.LBB20_23:                              //   in Loop: Header=BB20_25 Depth=1
	mov	x10, xzr
.LBB20_24:                              //   in Loop: Header=BB20_25 Depth=1
	cmp	x9, #8
	str	d0, [x19, x10, lsl #3]
	b.le	.LBB20_37
.LBB20_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_28 Depth 2
                                        //     Child Loop BB20_35 Depth 2
	ldr	d0, [x20, #-8]!
	sub	x9, x20, x19
	asr	x11, x9, #3
	ldr	d1, [x19]
	cmp	x11, #3
	str	d1, [x20]
	b.lt	.LBB20_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB20_25 Depth=1
	sub	x10, x11, #1
	mov	x13, xzr
	add	x10, x10, x10, lsr #63
	asr	x12, x10, #1
	b	.LBB20_28
.LBB20_27:                              // %select.end
                                        //   in Loop: Header=BB20_28 Depth=2
	ldr	d1, [x19, x10, lsl #3]
	cmp	x10, x12
	str	d1, [x19, x13, lsl #3]
	mov	x13, x10
	b.ge	.LBB20_31
.LBB20_28:                              //   Parent Loop BB20_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x19, x13, lsl #4
	add	x10, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x19, x10, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB20_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB20_28 Depth=2
	orr	x10, x14, #0x1
	b	.LBB20_27
.LBB20_30:                              //   in Loop: Header=BB20_25 Depth=1
	mov	x10, xzr
.LBB20_31:                              //   in Loop: Header=BB20_25 Depth=1
	tbnz	w9, #3, .LBB20_34
// %bb.32:                              //   in Loop: Header=BB20_25 Depth=1
	sub	x11, x11, #2
	cmp	x10, x11, asr #1
	b.ne	.LBB20_34
// %bb.33:                              //   in Loop: Header=BB20_25 Depth=1
	orr	x11, x8, x10, lsl #1
	ldr	d1, [x19, x11, lsl #3]
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b	.LBB20_35
.LBB20_34:                              //   in Loop: Header=BB20_25 Depth=1
	cbz	x10, .LBB20_24
.LBB20_35:                              //   Parent Loop BB20_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x19, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB20_24
// %bb.36:                              //   in Loop: Header=BB20_35 Depth=2
	cmp	x12, #1
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB20_35
	b	.LBB20_23
.LBB20_37:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldr	x25, [sp, #16]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #80             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w30
	.cfi_restore w29
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
	str	d8, [sp, #-80]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -80
	.cfi_remember_state
	sub	x8, x1, x0
	mov	x19, x1
	mov	x20, x0
	cmp	x8, #129
	b.lt	.LBB21_2
// %bb.1:
	add	x21, x20, #8
	mov	w22, #8                         // =0x8
	mov	x24, x20
	mov	x23, x21
	b	.LBB21_18
.LBB21_2:
	cmp	x20, x19
	b.eq	.LBB21_25
// %bb.3:
	add	x10, x20, #8
	cmp	x10, x19
	b.eq	.LBB21_25
// %bb.4:                               // %.preheader7
	mov	x21, x20
	b	.LBB21_9
.LBB21_5:                               //   in Loop: Header=BB21_9 Depth=1
	sub	x2, x21, x20
	asr	x8, x2, #3
	cmp	x8, #2
	b.lt	.LBB21_13
// %bb.6:                               //   in Loop: Header=BB21_9 Depth=1
	sub	x8, x9, x8, lsl #3
	mov	x1, x20
	add	x0, x8, #16
	bl	memmove
.LBB21_7:                               //   in Loop: Header=BB21_9 Depth=1
	mov	x8, x20
.LBB21_8:                               //   in Loop: Header=BB21_9 Depth=1
	add	x10, x21, #8
	str	d8, [x8]
	cmp	x10, x19
	b.eq	.LBB21_25
.LBB21_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_12 Depth 2
	ldr	d8, [x21, #8]
	ldr	d0, [x20]
	mov	x9, x21
	mov	x21, x10
	fcmp	d8, d0
	b.mi	.LBB21_5
// %bb.10:                              //   in Loop: Header=BB21_9 Depth=1
	ldr	d0, [x9]
	mov	x8, x21
	fcmp	d8, d0
	b.pl	.LBB21_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB21_9 Depth=1
	mov	x8, x21
.LBB21_12:                              //   Parent Loop BB21_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB21_12
	b	.LBB21_8
.LBB21_13:                              //   in Loop: Header=BB21_9 Depth=1
	cmp	x2, #8
	mov	x8, x20
	b.ne	.LBB21_8
// %bb.14:                              //   in Loop: Header=BB21_9 Depth=1
	str	d0, [x9, #8]
	b	.LBB21_7
.LBB21_15:                              //   in Loop: Header=BB21_18 Depth=1
	str	d0, [x8, #8]
.LBB21_16:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x8, x20
.LBB21_17:                              //   in Loop: Header=BB21_18 Depth=1
	add	x22, x22, #8
	add	x23, x23, #8
	str	d8, [x8]
	cmp	x22, #128
	b.eq	.LBB21_24
.LBB21_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_23 Depth 2
	mov	x8, x24
	add	x24, x20, x22
	ldr	d0, [x20]
	ldr	d8, [x24]
	fcmp	d8, d0
	b.pl	.LBB21_21
// %bb.19:                              //   in Loop: Header=BB21_18 Depth=1
	cmp	x22, #9
	b.lo	.LBB21_15
// %bb.20:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x0, x21
	mov	x1, x20
	mov	x2, x22
	bl	memmove
	b	.LBB21_16
.LBB21_21:                              //   in Loop: Header=BB21_18 Depth=1
	ldr	d0, [x8]
	mov	x8, x24
	fcmp	d8, d0
	b.pl	.LBB21_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB21_18 Depth=1
	mov	x8, x23
.LBB21_23:                              //   Parent Loop BB21_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB21_23
	b	.LBB21_17
.LBB21_24:
	add	x8, x20, #128
	cmp	x8, x19
	b.ne	.LBB21_27
.LBB21_25:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #80                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB21_26:                              //   in Loop: Header=BB21_27 Depth=1
	.cfi_restore_state
	add	x8, x8, #8
	str	d0, [x9]
	cmp	x8, x19
	b.eq	.LBB21_25
.LBB21_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_29 Depth 2
	ldp	d1, d0, [x8, #-8]
	mov	x9, x8
	fcmp	d0, d1
	b.pl	.LBB21_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB21_27 Depth=1
	mov	x9, x8
.LBB21_29:                              //   Parent Loop BB21_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x9]
	ldur	d1, [x9, #-16]
	sub	x9, x9, #8
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
	sub	x11, x1, x0
	asr	x8, x11, #3
	subs	x10, x8, #2
	b.ge	.LBB22_2
.LBB22_1:
	ret
.LBB22_2:
	sub	x8, x8, #1
	lsr	x9, x10, #1
	lsr	x8, x8, #1
	tbnz	w11, #3, .LBB22_18
// %bb.3:
	orr	x10, x10, #0x1
	mov	x12, x9
	b	.LBB22_6
.LBB22_4:                               //   in Loop: Header=BB22_6 Depth=1
	mov	x13, x12
.LBB22_5:                               //   in Loop: Header=BB22_6 Depth=1
	sub	x12, x11, #1
	str	d0, [x0, x13, lsl #3]
	cbz	x11, .LBB22_1
.LBB22_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_9 Depth 2
                                        //     Child Loop BB22_14 Depth 2
	ldr	d0, [x0, x12, lsl #3]
	mov	x11, x12
	cmp	x12, x8
	b.ge	.LBB22_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB22_6 Depth=1
	mov	x13, x11
	b	.LBB22_9
.LBB22_8:                               // %select.end
                                        //   in Loop: Header=BB22_9 Depth=2
	ldr	d1, [x0, x12, lsl #3]
	cmp	x12, x8
	str	d1, [x0, x13, lsl #3]
	mov	x13, x12
	b.ge	.LBB22_11
.LBB22_9:                               //   Parent Loop BB22_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x0, x13, lsl #4
	add	x12, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x0, x12, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB22_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB22_9 Depth=2
	orr	x12, x14, #0x1
	b	.LBB22_8
.LBB22_11:                              //   in Loop: Header=BB22_6 Depth=1
	cmp	x12, x9
	b.ne	.LBB22_13
// %bb.12:                              //   in Loop: Header=BB22_6 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	mov	x12, x10
	str	d1, [x0, x9, lsl #3]
.LBB22_13:                              //   in Loop: Header=BB22_6 Depth=1
	cmp	x12, x11
	b.le	.LBB22_4
.LBB22_14:                              //   Parent Loop BB22_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_4
// %bb.15:                              //   in Loop: Header=BB22_14 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.gt	.LBB22_14
	b	.LBB22_5
.LBB22_16:                              //   in Loop: Header=BB22_18 Depth=1
	mov	x11, x9
.LBB22_17:                              //   in Loop: Header=BB22_18 Depth=1
	sub	x9, x10, #1
	str	d0, [x0, x11, lsl #3]
	cbz	x10, .LBB22_1
.LBB22_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_21 Depth 2
                                        //     Child Loop BB22_24 Depth 2
	ldr	d0, [x0, x9, lsl #3]
	mov	x10, x9
	cmp	x9, x8
	mov	x11, x9
	b.ge	.LBB22_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB22_18 Depth=1
	mov	x11, x10
	b	.LBB22_21
.LBB22_20:                              // %select.end10
                                        //   in Loop: Header=BB22_21 Depth=2
	ldr	d1, [x0, x9, lsl #3]
	cmp	x9, x8
	str	d1, [x0, x11, lsl #3]
	mov	x11, x9
	b.ge	.LBB22_23
.LBB22_21:                              //   Parent Loop BB22_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x12, x11, #1
	add	x13, x0, x11, lsl #4
	add	x9, x12, #2
	ldr	d2, [x13, #8]
	ldr	d1, [x0, x9, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB22_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB22_21 Depth=2
	orr	x9, x12, #0x1
	b	.LBB22_20
.LBB22_23:                              //   in Loop: Header=BB22_18 Depth=1
	cmp	x9, x10
	b.le	.LBB22_16
.LBB22_24:                              //   Parent Loop BB22_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x11, x9, #1
	add	x11, x11, x11, lsr #63
	asr	x11, x11, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_16
// %bb.25:                              //   in Loop: Header=BB22_24 Depth=2
	cmp	x11, x10
	str	d1, [x0, x9, lsl #3]
	mov	x9, x11
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
	stp	x29, x30, [sp, #-80]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 80
	str	x25, [sp, #16]                  // 8-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 80
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -64
	.cfi_offset w30, -72
	.cfi_offset w29, -80
	sub	x25, x1, x0
	cmp	x25, #129
	b.lt	.LBB23_37
// %bb.1:
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x20, x3
	mov	x19, x0
	mov	x21, x2
	add	x23, x0, #8
	sub	x24, x8, x0
	b	.LBB23_3
.LBB23_2:                               //   in Loop: Header=BB23_3 Depth=1
	and	x3, x20, #0xff
	mov	x0, x22
	mov	x2, x21
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_T0_T1_
	cmp	x25, #128
	mov	x1, x22
	b.le	.LBB23_37
.LBB23_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_16 Depth 2
                                        //       Child Loop BB23_17 Depth 3
                                        //       Child Loop BB23_19 Depth 3
	cbz	x21, .LBB23_22
// %bb.4:                               //   in Loop: Header=BB23_3 Depth=1
	lsr	x8, x25, #4
	ldr	d1, [x19, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x19, x8, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB23_7
// %bb.5:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB23_9
// %bb.6:                               //   in Loop: Header=BB23_3 Depth=1
	ldr	d0, [x19]
	str	d2, [x19]
	str	d0, [x19, x8, lsl #3]
	b	.LBB23_15
.LBB23_7:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB23_11
// %bb.8:                               //   in Loop: Header=BB23_3 Depth=1
	ldr	d0, [x19]
	stp	d1, d0, [x19]
	b	.LBB23_15
.LBB23_9:                               //   in Loop: Header=BB23_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x19]
	b.pl	.LBB23_13
// %bb.10:                              //   in Loop: Header=BB23_3 Depth=1
	str	d0, [x19]
	stur	d2, [x1, #-8]
	b	.LBB23_15
.LBB23_11:                              //   in Loop: Header=BB23_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x19]
	b.pl	.LBB23_14
// %bb.12:                              //   in Loop: Header=BB23_3 Depth=1
	str	d0, [x19]
	stur	d1, [x1, #-8]
	b	.LBB23_15
.LBB23_13:                              //   in Loop: Header=BB23_3 Depth=1
	stp	d1, d2, [x19]
	b	.LBB23_15
.LBB23_14:                              //   in Loop: Header=BB23_3 Depth=1
	str	d2, [x19]
	str	d1, [x19, x8, lsl #3]
.LBB23_15:                              // %.preheader6
                                        //   in Loop: Header=BB23_3 Depth=1
	sub	x21, x21, #1
	mov	x8, x1
	mov	x9, x23
.LBB23_16:                              //   Parent Loop BB23_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB23_17 Depth 3
                                        //       Child Loop BB23_19 Depth 3
	ldr	d0, [x19]
	add	x25, x24, x9
.LBB23_17:                              //   Parent Loop BB23_3 Depth=1
                                        //     Parent Loop BB23_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x9], #8
	add	x25, x25, #8
	fcmp	d1, d0
	b.mi	.LBB23_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB23_16 Depth=2
	sub	x22, x9, #8
.LBB23_19:                              //   Parent Loop BB23_3 Depth=1
                                        //     Parent Loop BB23_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #-8]!
	fcmp	d0, d2
	b.mi	.LBB23_19
// %bb.20:                              //   in Loop: Header=BB23_16 Depth=2
	cmp	x22, x8
	b.hs	.LBB23_2
// %bb.21:                              //   in Loop: Header=BB23_16 Depth=2
	str	d2, [x22]
	str	d1, [x8]
	b	.LBB23_16
.LBB23_22:
	add	x2, x29, #28
	mov	x0, x19
	strb	w20, [x29, #28]
	mov	x20, x1
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_comp_iterISt4lessIdEEEEvT_S7_RT0_
	mov	w8, #1                          // =0x1
	b	.LBB23_25
.LBB23_23:                              //   in Loop: Header=BB23_25 Depth=1
	mov	x10, xzr
.LBB23_24:                              //   in Loop: Header=BB23_25 Depth=1
	cmp	x9, #8
	str	d0, [x19, x10, lsl #3]
	b.le	.LBB23_37
.LBB23_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_28 Depth 2
                                        //     Child Loop BB23_35 Depth 2
	ldr	d0, [x20, #-8]!
	sub	x9, x20, x19
	asr	x11, x9, #3
	ldr	d1, [x19]
	cmp	x11, #3
	str	d1, [x20]
	b.lt	.LBB23_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB23_25 Depth=1
	sub	x10, x11, #1
	mov	x13, xzr
	add	x10, x10, x10, lsr #63
	asr	x12, x10, #1
	b	.LBB23_28
.LBB23_27:                              // %select.end
                                        //   in Loop: Header=BB23_28 Depth=2
	ldr	d1, [x19, x10, lsl #3]
	cmp	x10, x12
	str	d1, [x19, x13, lsl #3]
	mov	x13, x10
	b.ge	.LBB23_31
.LBB23_28:                              //   Parent Loop BB23_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x19, x13, lsl #4
	add	x10, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x19, x10, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB23_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB23_28 Depth=2
	orr	x10, x14, #0x1
	b	.LBB23_27
.LBB23_30:                              //   in Loop: Header=BB23_25 Depth=1
	mov	x10, xzr
.LBB23_31:                              //   in Loop: Header=BB23_25 Depth=1
	tbnz	w9, #3, .LBB23_34
// %bb.32:                              //   in Loop: Header=BB23_25 Depth=1
	sub	x11, x11, #2
	cmp	x10, x11, asr #1
	b.ne	.LBB23_34
// %bb.33:                              //   in Loop: Header=BB23_25 Depth=1
	orr	x11, x8, x10, lsl #1
	ldr	d1, [x19, x11, lsl #3]
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b	.LBB23_35
.LBB23_34:                              //   in Loop: Header=BB23_25 Depth=1
	cbz	x10, .LBB23_24
.LBB23_35:                              //   Parent Loop BB23_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x19, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB23_24
// %bb.36:                              //   in Loop: Header=BB23_35 Depth=2
	cmp	x12, #1
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB23_35
	b	.LBB23_23
.LBB23_37:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldr	x25, [sp, #16]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #80             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w30
	.cfi_restore w29
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
	str	d8, [sp, #-80]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -80
	.cfi_remember_state
	sub	x8, x1, x0
	mov	x19, x1
	mov	x20, x0
	cmp	x8, #129
	b.lt	.LBB24_2
// %bb.1:
	add	x21, x20, #8
	mov	w22, #8                         // =0x8
	mov	x24, x20
	mov	x23, x21
	b	.LBB24_18
.LBB24_2:
	cmp	x20, x19
	b.eq	.LBB24_25
// %bb.3:
	add	x10, x20, #8
	cmp	x10, x19
	b.eq	.LBB24_25
// %bb.4:                               // %.preheader7
	mov	x21, x20
	b	.LBB24_9
.LBB24_5:                               //   in Loop: Header=BB24_9 Depth=1
	sub	x2, x21, x20
	asr	x8, x2, #3
	cmp	x8, #2
	b.lt	.LBB24_13
// %bb.6:                               //   in Loop: Header=BB24_9 Depth=1
	sub	x8, x9, x8, lsl #3
	mov	x1, x20
	add	x0, x8, #16
	bl	memmove
.LBB24_7:                               //   in Loop: Header=BB24_9 Depth=1
	mov	x8, x20
.LBB24_8:                               //   in Loop: Header=BB24_9 Depth=1
	add	x10, x21, #8
	str	d8, [x8]
	cmp	x10, x19
	b.eq	.LBB24_25
.LBB24_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_12 Depth 2
	ldr	d8, [x21, #8]
	ldr	d0, [x20]
	mov	x9, x21
	mov	x21, x10
	fcmp	d8, d0
	b.mi	.LBB24_5
// %bb.10:                              //   in Loop: Header=BB24_9 Depth=1
	ldr	d0, [x9]
	mov	x8, x21
	fcmp	d8, d0
	b.pl	.LBB24_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB24_9 Depth=1
	mov	x8, x21
.LBB24_12:                              //   Parent Loop BB24_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB24_12
	b	.LBB24_8
.LBB24_13:                              //   in Loop: Header=BB24_9 Depth=1
	cmp	x2, #8
	mov	x8, x20
	b.ne	.LBB24_8
// %bb.14:                              //   in Loop: Header=BB24_9 Depth=1
	str	d0, [x9, #8]
	b	.LBB24_7
.LBB24_15:                              //   in Loop: Header=BB24_18 Depth=1
	str	d0, [x8, #8]
.LBB24_16:                              //   in Loop: Header=BB24_18 Depth=1
	mov	x8, x20
.LBB24_17:                              //   in Loop: Header=BB24_18 Depth=1
	add	x22, x22, #8
	add	x23, x23, #8
	str	d8, [x8]
	cmp	x22, #128
	b.eq	.LBB24_24
.LBB24_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_23 Depth 2
	mov	x8, x24
	add	x24, x20, x22
	ldr	d0, [x20]
	ldr	d8, [x24]
	fcmp	d8, d0
	b.pl	.LBB24_21
// %bb.19:                              //   in Loop: Header=BB24_18 Depth=1
	cmp	x22, #9
	b.lo	.LBB24_15
// %bb.20:                              //   in Loop: Header=BB24_18 Depth=1
	mov	x0, x21
	mov	x1, x20
	mov	x2, x22
	bl	memmove
	b	.LBB24_16
.LBB24_21:                              //   in Loop: Header=BB24_18 Depth=1
	ldr	d0, [x8]
	mov	x8, x24
	fcmp	d8, d0
	b.pl	.LBB24_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB24_18 Depth=1
	mov	x8, x23
.LBB24_23:                              //   Parent Loop BB24_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB24_23
	b	.LBB24_17
.LBB24_24:
	add	x8, x20, #128
	cmp	x8, x19
	b.ne	.LBB24_27
.LBB24_25:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #80                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB24_26:                              //   in Loop: Header=BB24_27 Depth=1
	.cfi_restore_state
	add	x8, x8, #8
	str	d0, [x9]
	cmp	x8, x19
	b.eq	.LBB24_25
.LBB24_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_29 Depth 2
	ldp	d1, d0, [x8, #-8]
	mov	x9, x8
	fcmp	d0, d1
	b.pl	.LBB24_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB24_27 Depth=1
	mov	x9, x8
.LBB24_29:                              //   Parent Loop BB24_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x9]
	ldur	d1, [x9, #-16]
	sub	x9, x9, #8
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
	sub	x11, x1, x0
	asr	x8, x11, #3
	subs	x10, x8, #2
	b.ge	.LBB25_2
.LBB25_1:
	ret
.LBB25_2:
	sub	x8, x8, #1
	lsr	x9, x10, #1
	lsr	x8, x8, #1
	tbnz	w11, #3, .LBB25_18
// %bb.3:
	orr	x10, x10, #0x1
	mov	x12, x9
	b	.LBB25_6
.LBB25_4:                               //   in Loop: Header=BB25_6 Depth=1
	mov	x13, x12
.LBB25_5:                               //   in Loop: Header=BB25_6 Depth=1
	sub	x12, x11, #1
	str	d0, [x0, x13, lsl #3]
	cbz	x11, .LBB25_1
.LBB25_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_9 Depth 2
                                        //     Child Loop BB25_14 Depth 2
	ldr	d0, [x0, x12, lsl #3]
	mov	x11, x12
	cmp	x12, x8
	b.ge	.LBB25_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB25_6 Depth=1
	mov	x13, x11
	b	.LBB25_9
.LBB25_8:                               // %select.end
                                        //   in Loop: Header=BB25_9 Depth=2
	ldr	d1, [x0, x12, lsl #3]
	cmp	x12, x8
	str	d1, [x0, x13, lsl #3]
	mov	x13, x12
	b.ge	.LBB25_11
.LBB25_9:                               //   Parent Loop BB25_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x0, x13, lsl #4
	add	x12, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x0, x12, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB25_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB25_9 Depth=2
	orr	x12, x14, #0x1
	b	.LBB25_8
.LBB25_11:                              //   in Loop: Header=BB25_6 Depth=1
	cmp	x12, x9
	b.ne	.LBB25_13
// %bb.12:                              //   in Loop: Header=BB25_6 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	mov	x12, x10
	str	d1, [x0, x9, lsl #3]
.LBB25_13:                              //   in Loop: Header=BB25_6 Depth=1
	cmp	x12, x11
	b.le	.LBB25_4
.LBB25_14:                              //   Parent Loop BB25_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB25_4
// %bb.15:                              //   in Loop: Header=BB25_14 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.gt	.LBB25_14
	b	.LBB25_5
.LBB25_16:                              //   in Loop: Header=BB25_18 Depth=1
	mov	x11, x9
.LBB25_17:                              //   in Loop: Header=BB25_18 Depth=1
	sub	x9, x10, #1
	str	d0, [x0, x11, lsl #3]
	cbz	x10, .LBB25_1
.LBB25_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_21 Depth 2
                                        //     Child Loop BB25_24 Depth 2
	ldr	d0, [x0, x9, lsl #3]
	mov	x10, x9
	cmp	x9, x8
	mov	x11, x9
	b.ge	.LBB25_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB25_18 Depth=1
	mov	x11, x10
	b	.LBB25_21
.LBB25_20:                              // %select.end10
                                        //   in Loop: Header=BB25_21 Depth=2
	ldr	d1, [x0, x9, lsl #3]
	cmp	x9, x8
	str	d1, [x0, x11, lsl #3]
	mov	x11, x9
	b.ge	.LBB25_23
.LBB25_21:                              //   Parent Loop BB25_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x12, x11, #1
	add	x13, x0, x11, lsl #4
	add	x9, x12, #2
	ldr	d2, [x13, #8]
	ldr	d1, [x0, x9, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB25_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB25_21 Depth=2
	orr	x9, x12, #0x1
	b	.LBB25_20
.LBB25_23:                              //   in Loop: Header=BB25_18 Depth=1
	cmp	x9, x10
	b.le	.LBB25_16
.LBB25_24:                              //   Parent Loop BB25_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x11, x9, #1
	add	x11, x11, x11, lsr #63
	asr	x11, x11, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB25_16
// %bb.25:                              //   in Loop: Header=BB25_24 Depth=2
	cmp	x11, x10
	str	d1, [x0, x9, lsl #3]
	mov	x9, x11
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
	sub	sp, sp, #80
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	sub	x24, x1, x0
	cmp	x24, #129
	b.lt	.LBB26_37
// %bb.1:
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x19, x0
	mov	x20, x2
	add	x22, x0, #8
	sub	x23, x8, x0
	b	.LBB26_3
.LBB26_2:                               //   in Loop: Header=BB26_3 Depth=1
	mov	x0, x21
	mov	x2, x20
	bl	_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_
	cmp	x24, #128
	mov	x1, x21
	b.le	.LBB26_37
.LBB26_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_16 Depth 2
                                        //       Child Loop BB26_17 Depth 3
                                        //       Child Loop BB26_19 Depth 3
	cbz	x20, .LBB26_22
// %bb.4:                               //   in Loop: Header=BB26_3 Depth=1
	lsr	x8, x24, #4
	ldr	d1, [x19, #8]
	ldur	d0, [x1, #-8]
	ldr	d2, [x19, x8, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB26_7
// %bb.5:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d2, d0
	b.pl	.LBB26_9
// %bb.6:                               //   in Loop: Header=BB26_3 Depth=1
	ldr	d0, [x19]
	str	d2, [x19]
	str	d0, [x19, x8, lsl #3]
	b	.LBB26_15
.LBB26_7:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d1, d0
	b.pl	.LBB26_11
// %bb.8:                               //   in Loop: Header=BB26_3 Depth=1
	ldr	d0, [x19]
	stp	d1, d0, [x19]
	b	.LBB26_15
.LBB26_9:                               //   in Loop: Header=BB26_3 Depth=1
	fcmp	d1, d0
	ldr	d2, [x19]
	b.pl	.LBB26_13
// %bb.10:                              //   in Loop: Header=BB26_3 Depth=1
	str	d0, [x19]
	stur	d2, [x1, #-8]
	b	.LBB26_15
.LBB26_11:                              //   in Loop: Header=BB26_3 Depth=1
	fcmp	d2, d0
	ldr	d1, [x19]
	b.pl	.LBB26_14
// %bb.12:                              //   in Loop: Header=BB26_3 Depth=1
	str	d0, [x19]
	stur	d1, [x1, #-8]
	b	.LBB26_15
.LBB26_13:                              //   in Loop: Header=BB26_3 Depth=1
	stp	d1, d2, [x19]
	b	.LBB26_15
.LBB26_14:                              //   in Loop: Header=BB26_3 Depth=1
	str	d2, [x19]
	str	d1, [x19, x8, lsl #3]
.LBB26_15:                              // %.preheader6
                                        //   in Loop: Header=BB26_3 Depth=1
	sub	x20, x20, #1
	mov	x8, x1
	mov	x9, x22
.LBB26_16:                              //   Parent Loop BB26_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB26_17 Depth 3
                                        //       Child Loop BB26_19 Depth 3
	ldr	d0, [x19]
	add	x24, x23, x9
.LBB26_17:                              //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x9], #8
	add	x24, x24, #8
	fcmp	d1, d0
	b.mi	.LBB26_17
// %bb.18:                              // %.preheader5
                                        //   in Loop: Header=BB26_16 Depth=2
	sub	x21, x9, #8
.LBB26_19:                              //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_16 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x8, #-8]!
	fcmp	d0, d2
	b.mi	.LBB26_19
// %bb.20:                              //   in Loop: Header=BB26_16 Depth=2
	cmp	x21, x8
	b.hs	.LBB26_2
// %bb.21:                              //   in Loop: Header=BB26_16 Depth=2
	str	d2, [x21]
	str	d1, [x8]
	b	.LBB26_16
.LBB26_22:
	sub	x2, x29, #1
	mov	x0, x19
	mov	x20, x1
	bl	_ZSt11__make_heapIPdN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_RT0_
	mov	w8, #1                          // =0x1
	b	.LBB26_25
.LBB26_23:                              //   in Loop: Header=BB26_25 Depth=1
	mov	x10, xzr
.LBB26_24:                              //   in Loop: Header=BB26_25 Depth=1
	cmp	x9, #8
	str	d0, [x19, x10, lsl #3]
	b.le	.LBB26_37
.LBB26_25:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_28 Depth 2
                                        //     Child Loop BB26_35 Depth 2
	ldr	d0, [x20, #-8]!
	sub	x9, x20, x19
	asr	x11, x9, #3
	ldr	d1, [x19]
	cmp	x11, #3
	str	d1, [x20]
	b.lt	.LBB26_30
// %bb.26:                              // %.preheader2
                                        //   in Loop: Header=BB26_25 Depth=1
	sub	x10, x11, #1
	mov	x13, xzr
	add	x10, x10, x10, lsr #63
	asr	x12, x10, #1
	b	.LBB26_28
.LBB26_27:                              // %select.end
                                        //   in Loop: Header=BB26_28 Depth=2
	ldr	d1, [x19, x10, lsl #3]
	cmp	x10, x12
	str	d1, [x19, x13, lsl #3]
	mov	x13, x10
	b.ge	.LBB26_31
.LBB26_28:                              //   Parent Loop BB26_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x19, x13, lsl #4
	add	x10, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x19, x10, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB26_27
// %bb.29:                              // %select.true.sink
                                        //   in Loop: Header=BB26_28 Depth=2
	orr	x10, x14, #0x1
	b	.LBB26_27
.LBB26_30:                              //   in Loop: Header=BB26_25 Depth=1
	mov	x10, xzr
.LBB26_31:                              //   in Loop: Header=BB26_25 Depth=1
	tbnz	w9, #3, .LBB26_34
// %bb.32:                              //   in Loop: Header=BB26_25 Depth=1
	sub	x11, x11, #2
	cmp	x10, x11, asr #1
	b.ne	.LBB26_34
// %bb.33:                              //   in Loop: Header=BB26_25 Depth=1
	orr	x11, x8, x10, lsl #1
	ldr	d1, [x19, x11, lsl #3]
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b	.LBB26_35
.LBB26_34:                              //   in Loop: Header=BB26_25 Depth=1
	cbz	x10, .LBB26_24
.LBB26_35:                              //   Parent Loop BB26_25 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x19, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB26_24
// %bb.36:                              //   in Loop: Header=BB26_35 Depth=2
	cmp	x12, #1
	str	d1, [x19, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB26_35
	b	.LBB26_23
.LBB26_37:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #80
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w30
	.cfi_restore w29
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
	str	d8, [sp, #-80]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 80
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #32]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #48]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #64]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 64
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w30, -56
	.cfi_offset w29, -64
	.cfi_offset b8, -80
	.cfi_remember_state
	sub	x8, x1, x0
	mov	x19, x1
	mov	x20, x0
	cmp	x8, #129
	b.lt	.LBB27_2
// %bb.1:
	add	x21, x20, #8
	mov	w22, #8                         // =0x8
	mov	x24, x20
	mov	x23, x21
	b	.LBB27_18
.LBB27_2:
	cmp	x20, x19
	b.eq	.LBB27_25
// %bb.3:
	add	x10, x20, #8
	cmp	x10, x19
	b.eq	.LBB27_25
// %bb.4:                               // %.preheader7
	mov	x21, x20
	b	.LBB27_9
.LBB27_5:                               //   in Loop: Header=BB27_9 Depth=1
	sub	x2, x21, x20
	asr	x8, x2, #3
	cmp	x8, #2
	b.lt	.LBB27_13
// %bb.6:                               //   in Loop: Header=BB27_9 Depth=1
	sub	x8, x9, x8, lsl #3
	mov	x1, x20
	add	x0, x8, #16
	bl	memmove
.LBB27_7:                               //   in Loop: Header=BB27_9 Depth=1
	mov	x8, x20
.LBB27_8:                               //   in Loop: Header=BB27_9 Depth=1
	add	x10, x21, #8
	str	d8, [x8]
	cmp	x10, x19
	b.eq	.LBB27_25
.LBB27_9:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_12 Depth 2
	ldr	d8, [x21, #8]
	ldr	d0, [x20]
	mov	x9, x21
	mov	x21, x10
	fcmp	d8, d0
	b.mi	.LBB27_5
// %bb.10:                              //   in Loop: Header=BB27_9 Depth=1
	ldr	d0, [x9]
	mov	x8, x21
	fcmp	d8, d0
	b.pl	.LBB27_8
// %bb.11:                              // %.preheader5
                                        //   in Loop: Header=BB27_9 Depth=1
	mov	x8, x21
.LBB27_12:                              //   Parent Loop BB27_9 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB27_12
	b	.LBB27_8
.LBB27_13:                              //   in Loop: Header=BB27_9 Depth=1
	cmp	x2, #8
	mov	x8, x20
	b.ne	.LBB27_8
// %bb.14:                              //   in Loop: Header=BB27_9 Depth=1
	str	d0, [x9, #8]
	b	.LBB27_7
.LBB27_15:                              //   in Loop: Header=BB27_18 Depth=1
	str	d0, [x8, #8]
.LBB27_16:                              //   in Loop: Header=BB27_18 Depth=1
	mov	x8, x20
.LBB27_17:                              //   in Loop: Header=BB27_18 Depth=1
	add	x22, x22, #8
	add	x23, x23, #8
	str	d8, [x8]
	cmp	x22, #128
	b.eq	.LBB27_24
.LBB27_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_23 Depth 2
	mov	x8, x24
	add	x24, x20, x22
	ldr	d0, [x20]
	ldr	d8, [x24]
	fcmp	d8, d0
	b.pl	.LBB27_21
// %bb.19:                              //   in Loop: Header=BB27_18 Depth=1
	cmp	x22, #9
	b.lo	.LBB27_15
// %bb.20:                              //   in Loop: Header=BB27_18 Depth=1
	mov	x0, x21
	mov	x1, x20
	mov	x2, x22
	bl	memmove
	b	.LBB27_16
.LBB27_21:                              //   in Loop: Header=BB27_18 Depth=1
	ldr	d0, [x8]
	mov	x8, x24
	fcmp	d8, d0
	b.pl	.LBB27_17
// %bb.22:                              // %.preheader3
                                        //   in Loop: Header=BB27_18 Depth=1
	mov	x8, x23
.LBB27_23:                              //   Parent Loop BB27_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d0, [x8]
	ldur	d0, [x8, #-16]
	sub	x8, x8, #8
	fcmp	d8, d0
	b.mi	.LBB27_23
	b	.LBB27_17
.LBB27_24:
	add	x8, x20, #128
	cmp	x8, x19
	b.ne	.LBB27_27
.LBB27_25:
	.cfi_def_cfa wsp, 80
	ldp	x20, x19, [sp, #64]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #48]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #80                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.LBB27_26:                              //   in Loop: Header=BB27_27 Depth=1
	.cfi_restore_state
	add	x8, x8, #8
	str	d0, [x9]
	cmp	x8, x19
	b.eq	.LBB27_25
.LBB27_27:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_29 Depth 2
	ldp	d1, d0, [x8, #-8]
	mov	x9, x8
	fcmp	d0, d1
	b.pl	.LBB27_26
// %bb.28:                              // %.preheader
                                        //   in Loop: Header=BB27_27 Depth=1
	mov	x9, x8
.LBB27_29:                              //   Parent Loop BB27_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	str	d1, [x9]
	ldur	d1, [x9, #-16]
	sub	x9, x9, #8
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
	sub	x11, x1, x0
	asr	x8, x11, #3
	subs	x10, x8, #2
	b.ge	.LBB28_2
.LBB28_1:
	ret
.LBB28_2:
	sub	x8, x8, #1
	lsr	x9, x10, #1
	lsr	x8, x8, #1
	tbnz	w11, #3, .LBB28_18
// %bb.3:
	orr	x10, x10, #0x1
	mov	x12, x9
	b	.LBB28_6
.LBB28_4:                               //   in Loop: Header=BB28_6 Depth=1
	mov	x13, x12
.LBB28_5:                               //   in Loop: Header=BB28_6 Depth=1
	sub	x12, x11, #1
	str	d0, [x0, x13, lsl #3]
	cbz	x11, .LBB28_1
.LBB28_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_9 Depth 2
                                        //     Child Loop BB28_14 Depth 2
	ldr	d0, [x0, x12, lsl #3]
	mov	x11, x12
	cmp	x12, x8
	b.ge	.LBB28_11
// %bb.7:                               // %.preheader1
                                        //   in Loop: Header=BB28_6 Depth=1
	mov	x13, x11
	b	.LBB28_9
.LBB28_8:                               // %select.end
                                        //   in Loop: Header=BB28_9 Depth=2
	ldr	d1, [x0, x12, lsl #3]
	cmp	x12, x8
	str	d1, [x0, x13, lsl #3]
	mov	x13, x12
	b.ge	.LBB28_11
.LBB28_9:                               //   Parent Loop BB28_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x14, x13, #1
	add	x15, x0, x13, lsl #4
	add	x12, x14, #2
	ldr	d2, [x15, #8]
	ldr	d1, [x0, x12, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB28_8
// %bb.10:                              // %select.true.sink
                                        //   in Loop: Header=BB28_9 Depth=2
	orr	x12, x14, #0x1
	b	.LBB28_8
.LBB28_11:                              //   in Loop: Header=BB28_6 Depth=1
	cmp	x12, x9
	b.ne	.LBB28_13
// %bb.12:                              //   in Loop: Header=BB28_6 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	mov	x12, x10
	str	d1, [x0, x9, lsl #3]
.LBB28_13:                              //   in Loop: Header=BB28_6 Depth=1
	cmp	x12, x11
	b.le	.LBB28_4
.LBB28_14:                              //   Parent Loop BB28_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB28_4
// %bb.15:                              //   in Loop: Header=BB28_14 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.gt	.LBB28_14
	b	.LBB28_5
.LBB28_16:                              //   in Loop: Header=BB28_18 Depth=1
	mov	x11, x9
.LBB28_17:                              //   in Loop: Header=BB28_18 Depth=1
	sub	x9, x10, #1
	str	d0, [x0, x11, lsl #3]
	cbz	x10, .LBB28_1
.LBB28_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_21 Depth 2
                                        //     Child Loop BB28_24 Depth 2
	ldr	d0, [x0, x9, lsl #3]
	mov	x10, x9
	cmp	x9, x8
	mov	x11, x9
	b.ge	.LBB28_17
// %bb.19:                              // %.preheader7
                                        //   in Loop: Header=BB28_18 Depth=1
	mov	x11, x10
	b	.LBB28_21
.LBB28_20:                              // %select.end10
                                        //   in Loop: Header=BB28_21 Depth=2
	ldr	d1, [x0, x9, lsl #3]
	cmp	x9, x8
	str	d1, [x0, x11, lsl #3]
	mov	x11, x9
	b.ge	.LBB28_23
.LBB28_21:                              //   Parent Loop BB28_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	lsl	x12, x11, #1
	add	x13, x0, x11, lsl #4
	add	x9, x12, #2
	ldr	d2, [x13, #8]
	ldr	d1, [x0, x9, lsl #3]
	fcmp	d1, d2
	b.pl	.LBB28_20
// %bb.22:                              // %select.true.sink11
                                        //   in Loop: Header=BB28_21 Depth=2
	orr	x9, x12, #0x1
	b	.LBB28_20
.LBB28_23:                              //   in Loop: Header=BB28_18 Depth=1
	cmp	x9, x10
	b.le	.LBB28_16
.LBB28_24:                              //   Parent Loop BB28_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x11, x9, #1
	add	x11, x11, x11, lsr #63
	asr	x11, x11, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB28_16
// %bb.25:                              //   in Loop: Header=BB28_24 Depth=2
	cmp	x11, x10
	str	d1, [x0, x9, lsl #3]
	mov	x9, x11
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
