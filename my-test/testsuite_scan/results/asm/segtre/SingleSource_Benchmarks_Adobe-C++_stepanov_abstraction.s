	.file	"stepanov_abstraction.cpp"
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
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	stp	x29, x30, [sp, #-96]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 96
	stp	x28, x27, [sp, #16]             // 16-byte Folded Spill
	stp	x26, x25, [sp, #32]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #48]             // 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             // 16-byte Folded Spill
	mov	x29, sp
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
	cmp	w1, #2
	adrp	x4, iterations
	adrp	x13, init_value
	b.lt	.LBB5_3
// %bb.1:
	ldr	x0, [x1, #8]
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	mov	x2, x0
	cmp	w1, #2
	str	w2, [x4, :lo12:iterations]
	b.eq	.LBB5_3
// %bb.2:
	ldr	x0, [x0, #16]
	mov	x1, xzr
	bl	strtod
	str	d0, [x13, :lo12:init_value]
.LBB5_3:
	ldr	d0, [x13, :lo12:init_value]
	fcvtzs	w0, d0
	add	w0, w0, #123
	bl	srand
	adrp	x0, dpb
	adrp	x1, dpe
	ldr	d0, [x13, :lo12:init_value]
	ldr	x16, [x0, :lo12:dpb]
	ldr	x17, [x1, :lo12:dpe]
	cmp	x16, x17
	b.eq	.LBB5_10
// %bb.4:
	sub	x2, x17, x16
	sub	x3, x2, #8
	mov	x2, x16
	cmp	x3, #24
	b.lo	.LBB5_8
// %bb.5:
	lsr	x2, x3, #3
	dup	v1.2d, v0.d[0]
	add	x6, x16, #16
	add	x3, x2, #1
	and	x5, x3, #0x3ffffffffffffffc
	add	x2, x16, x5, lsl #3
	mov	x7, x5
.LBB5_6:                                // =>This Inner Loop Header: Depth=1
	subs	x7, x7, #4
	stp	q1, q1, [x6, #-16]
	add	x6, x6, #32
	b.ne	.LBB5_6
// %bb.7:
	cmp	x3, x5
	b.eq	.LBB5_9
.LBB5_8:                                // =>This Inner Loop Header: Depth=1
	str	d0, [x2], #8
	cmp	x2, x17
	b.ne	.LBB5_8
.LBB5_9:
	ldr	d0, [x13, :lo12:init_value]
.LBB5_10:
	adrp	x2, DVpb
	adrp	x3, DVpe
	ldr	x9, [x2, :lo12:DVpb]
	ldr	x5, [x3, :lo12:DVpe]
	cmp	x9, x5
	b.eq	.LBB5_17
// %bb.11:
	sub	x6, x5, x9
	sub	x7, x6, #8
	mov	x6, x9
	cmp	x7, #24
	b.lo	.LBB5_15
// %bb.12:
	lsr	x6, x7, #3
	dup	v1.2d, v0.d[0]
	add	x7, x6, #1
	and	x8, x7, #0x3ffffffffffffffc
	add	x6, x9, x8, lsl #3
	add	x9, x9, #16
	mov	x10, x8
.LBB5_13:                               // =>This Inner Loop Header: Depth=1
	subs	x10, x10, #4
	stp	q1, q1, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB5_13
// %bb.14:
	cmp	x7, x8
	b.eq	.LBB5_16
.LBB5_15:                               // =>This Inner Loop Header: Depth=1
	str	d0, [x6], #8
	cmp	x6, x5
	b.ne	.LBB5_15
.LBB5_16:
	ldr	d0, [x13, :lo12:init_value]
.LBB5_17:
	adrp	x5, DV10pb
	adrp	x6, DV10pe
	ldr	x11, [x5, :lo12:DV10pb]
	ldr	x7, [x6, :lo12:DV10pe]
	cmp	x11, x7
	b.eq	.LBB5_23
// %bb.18:
	sub	x8, x7, x11
	sub	x9, x8, #8
	mov	x8, x11
	cmp	x9, #24
	b.lo	.LBB5_22
// %bb.19:
	lsr	x8, x9, #3
	dup	v1.2d, v0.d[0]
	add	x9, x8, #1
	and	x10, x9, #0x3ffffffffffffffc
	add	x8, x11, x10, lsl #3
	add	x11, x11, #16
	mov	x12, x10
.LBB5_20:                               // =>This Inner Loop Header: Depth=1
	subs	x12, x12, #4
	stp	q1, q1, [x11, #-16]
	add	x11, x11, #32
	b.ne	.LBB5_20
// %bb.21:
	cmp	x9, x10
	b.eq	.LBB5_23
.LBB5_22:                               // =>This Inner Loop Header: Depth=1
	str	d0, [x8], #8
	cmp	x8, x7
	b.ne	.LBB5_22
.LBB5_23:
	ldr	w15, [x4, :lo12:iterations]
	adrp	x12, dPb
	adrp	x11, dPe
	adrp	x9, DVPb
	adrp	x10, DVPe
	adrp	x7, DV10Pb
	cmp	w15, #1
	adrp	x8, DV10Pe
	b.lt	.LBB5_95
// %bb.24:
	cmp	x16, x17
	adrp	x14, current_test
	b.eq	.LBB5_31
// %bb.25:                              // %.preheader23
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_27
.LBB5_26:                               //   in Loop: Header=BB5_27 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_35
.LBB5_27:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_28 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_28:                               //   Parent Loop BB5_27 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_28
// %bb.29:                              //   in Loop: Header=BB5_27 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_26
// %bb.30:                              //   in Loop: Header=BB5_27 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_26
.LBB5_31:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_33
.LBB5_32:                               //   in Loop: Header=BB5_33 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_35
.LBB5_33:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_32
// %bb.34:                              //   in Loop: Header=BB5_33 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_32
.LBB5_35:
	cmp	w15, #1
	b.lt	.LBB5_95
// %bb.36:
	ldr	x16, [x12, :lo12:dPb]
	ldr	x17, [x11, :lo12:dPe]
	cmp	x16, x17
	b.eq	.LBB5_43
// %bb.37:                              // %.preheader20
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_39
.LBB5_38:                               //   in Loop: Header=BB5_39 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_47
.LBB5_39:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_40 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_40:                               //   Parent Loop BB5_39 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_40
// %bb.41:                              //   in Loop: Header=BB5_39 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_38
// %bb.42:                              //   in Loop: Header=BB5_39 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_38
.LBB5_43:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_45
.LBB5_44:                               //   in Loop: Header=BB5_45 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_47
.LBB5_45:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_44
// %bb.46:                              //   in Loop: Header=BB5_45 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_44
.LBB5_47:
	cmp	w15, #1
	b.lt	.LBB5_95
// %bb.48:
	ldr	x16, [x2, :lo12:DVpb]
	ldr	x17, [x3, :lo12:DVpe]
	cmp	x16, x17
	b.eq	.LBB5_55
// %bb.49:                              // %.preheader17
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_51
.LBB5_50:                               //   in Loop: Header=BB5_51 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_59
.LBB5_51:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_52 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_52:                               //   Parent Loop BB5_51 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_52
// %bb.53:                              //   in Loop: Header=BB5_51 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_50
// %bb.54:                              //   in Loop: Header=BB5_51 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_50
.LBB5_55:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_57
.LBB5_56:                               //   in Loop: Header=BB5_57 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_59
.LBB5_57:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_56
// %bb.58:                              //   in Loop: Header=BB5_57 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_56
.LBB5_59:
	cmp	w15, #1
	b.lt	.LBB5_95
// %bb.60:
	ldr	x16, [x9, :lo12:DVPb]
	ldr	x17, [x10, :lo12:DVPe]
	cmp	x16, x17
	b.eq	.LBB5_67
// %bb.61:                              // %.preheader14
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_63
.LBB5_62:                               //   in Loop: Header=BB5_63 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_71
.LBB5_63:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_64 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_64:                               //   Parent Loop BB5_63 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_64
// %bb.65:                              //   in Loop: Header=BB5_63 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_62
// %bb.66:                              //   in Loop: Header=BB5_63 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_62
.LBB5_67:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_69
.LBB5_68:                               //   in Loop: Header=BB5_69 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_71
.LBB5_69:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_68
// %bb.70:                              //   in Loop: Header=BB5_69 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_68
.LBB5_71:
	cmp	w15, #1
	b.lt	.LBB5_95
// %bb.72:
	ldr	x16, [x5, :lo12:DV10pb]
	ldr	x17, [x6, :lo12:DV10pe]
	cmp	x16, x17
	b.eq	.LBB5_79
// %bb.73:                              // %.preheader11
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_75
.LBB5_74:                               //   in Loop: Header=BB5_75 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_83
.LBB5_75:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_76 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_76:                               //   Parent Loop BB5_75 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_76
// %bb.77:                              //   in Loop: Header=BB5_75 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_74
// %bb.78:                              //   in Loop: Header=BB5_75 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_74
.LBB5_79:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_81
.LBB5_80:                               //   in Loop: Header=BB5_81 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_83
.LBB5_81:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_80
// %bb.82:                              //   in Loop: Header=BB5_81 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_80
.LBB5_83:
	cmp	w15, #1
	b.lt	.LBB5_95
// %bb.84:
	ldr	x16, [x7, :lo12:DV10Pb]
	ldr	x17, [x8, :lo12:DV10Pe]
	cmp	x16, x17
	b.eq	.LBB5_91
// %bb.85:                              // %.preheader8
	mov	x19, #70368744177664            // =0x400000000000
	mov	w18, wzr
	movk	x19, #16543, lsl #48
	fmov	d0, x19
	adrp	x19, .L.str.50
	add	x19, x19, :lo12:.L.str.50
	b	.LBB5_87
.LBB5_86:                               //   in Loop: Header=BB5_87 Depth=1
	add	w18, w18, #1
	cmp	w18, w15
	b.ge	.LBB5_95
.LBB5_87:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_88 Depth 2
	movi	d1, #0000000000000000
	mov	x20, x16
.LBB5_88:                               //   Parent Loop BB5_87 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x20], #8
	cmp	x20, x17
	fadd	d1, d1, d2
	b.ne	.LBB5_88
// %bb.89:                              //   in Loop: Header=BB5_87 Depth=1
	ldr	d2, [x13, :lo12:init_value]
	fmul	d2, d2, d0
	fcmp	d1, d2
	b.eq	.LBB5_86
// %bb.90:                              //   in Loop: Header=BB5_87 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x19
	bl	printf
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_86
.LBB5_91:
	mov	x17, #70368744177664            // =0x400000000000
	ldr	d1, [x13, :lo12:init_value]
	mov	w16, wzr
	movk	x17, #16543, lsl #48
	fmov	d0, x17
	adrp	x17, .L.str.50
	add	x17, x17, :lo12:.L.str.50
	b	.LBB5_93
.LBB5_92:                               //   in Loop: Header=BB5_93 Depth=1
	add	w16, w16, #1
	cmp	w16, w15
	b.ge	.LBB5_95
.LBB5_93:                               // =>This Inner Loop Header: Depth=1
	fmul	d2, d1, d0
	fcmp	d2, #0.0
	b.eq	.LBB5_92
// %bb.94:                              //   in Loop: Header=BB5_93 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x17
	bl	printf
	ldr	d1, [x13, :lo12:init_value]
	ldr	w15, [x4, :lo12:iterations]
	b	.LBB5_92
.LBB5_95:
	mov	w13, #19923                     // =0x4dd3
	adrp	x14, dMpb
	movk	w13, #4194, lsl #16
	ldr	x17, [x14, :lo12:dMpb]
	smull	x13, w15, w13
	adrp	x15, dMpe
	ldr	x18, [x15, :lo12:dMpe]
	asr	x13, x13, #39
	cmp	x17, x18
	add	w13, w13, w13, lsr #31
	str	w13, [x4, :lo12:iterations]
	b.eq	.LBB5_98
.LBB5_96:                               // =>This Inner Loop Header: Depth=1
	bl	rand
	scvtf	d0, w0
	str	d0, [x17], #8
	cmp	x17, x18
	b.ne	.LBB5_96
// %bb.97:
	ldr	x17, [x14, :lo12:dMpb]
	ldr	x18, [x15, :lo12:dMpe]
.LBB5_98:
	cmp	x17, x18
	adrp	x16, DVMpb
	adrp	x13, DV10Mpb
	b.eq	.LBB5_113
// %bb.99:
	sub	x19, x18, x17
	ldr	x22, [x16, :lo12:DVMpb]
	sub	x20, x19, #8
	lsr	x19, x20, #3
	cmp	x20, #24
	b.lo	.LBB5_104
// %bb.100:
	sub	x21, x22, x17
	cmp	x21, #32
	mov	x21, x17
	b.lo	.LBB5_105
// %bb.101:
	add	x23, x19, #1
	add	x26, x17, #16
	and	x24, x23, #0x3ffffffffffffffc
	lsl	x21, x24, #3
	mov	x27, x24
	add	x25, x22, x21
	add	x21, x17, x21
	add	x22, x22, #16
.LBB5_102:                              // =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x26, #-16]
	subs	x27, x27, #4
	add	x26, x26, #32
	stp	q0, q1, [x22, #-16]
	add	x22, x22, #32
	b.ne	.LBB5_102
// %bb.103:
	cmp	x23, x24
	mov	x22, x25
	b.ne	.LBB5_105
	b	.LBB5_106
.LBB5_104:
	mov	x21, x17
.LBB5_105:                              // =>This Inner Loop Header: Depth=1
	ldr	x23, [x21], #8
	cmp	x21, x18
	str	x23, [x22], #8
	b.ne	.LBB5_105
.LBB5_106:
	ldr	x21, [x13, :lo12:DV10Mpb]
	cmp	x20, #24
	b.lo	.LBB5_111
// %bb.107:
	sub	x20, x21, x17
	cmp	x20, #32
	mov	x20, x17
	b.lo	.LBB5_112
// %bb.108:
	add	x19, x19, #1
	add	x24, x17, #16
	and	x22, x19, #0x3ffffffffffffffc
	lsl	x20, x22, #3
	mov	x25, x22
	add	x23, x21, x20
	add	x20, x17, x20
	add	x21, x21, #16
.LBB5_109:                              // =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x24, #-16]
	subs	x25, x25, #4
	add	x24, x24, #32
	stp	q0, q1, [x21, #-16]
	add	x21, x21, #32
	b.ne	.LBB5_109
// %bb.110:
	cmp	x19, x22
	mov	x21, x23
	b.ne	.LBB5_112
	b	.LBB5_113
.LBB5_111:
	mov	x20, x17
.LBB5_112:                              // =>This Inner Loop Header: Depth=1
	ldr	x19, [x20], #8
	cmp	x20, x18
	str	x19, [x21], #8
	b.ne	.LBB5_112
.LBB5_113:
	movi	d0, #0000000000000000
	ldr	x2, [x0, :lo12:dpb]
	ldr	x3, [x1, :lo12:dpe]
	adrp	x4, .L.str.32
	add	x4, x4, :lo12:.L.str.32
	mov	x0, x17
	mov	x1, x18
	bl	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	movi	d0, #0000000000000000
	adrp	x17, dMPb
	adrp	x18, dMPe
	ldr	x0, [x17, :lo12:dMPb]
	ldr	x1, [x18, :lo12:dMPe]
	ldr	x2, [x12, :lo12:dPb]
	ldr	x3, [x11, :lo12:dPe]
	adrp	x4, .L.str.33
	add	x4, x4, :lo12:.L.str.33
	bl	_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	movi	d0, #0000000000000000
	adrp	x19, DVMpe
	ldr	x0, [x16, :lo12:DVMpb]
	ldr	x1, [x19, :lo12:DVMpe]
	ldr	x2, [x2, :lo12:DVpb]
	ldr	x3, [x3, :lo12:DVpe]
	adrp	x4, .L.str.34
	add	x4, x4, :lo12:.L.str.34
	bl	_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	movi	d0, #0000000000000000
	adrp	x20, DVMPb
	adrp	x21, DVMPe
	ldr	x0, [x20, :lo12:DVMPb]
	ldr	x1, [x21, :lo12:DVMPe]
	ldr	x2, [x9, :lo12:DVPb]
	ldr	x3, [x10, :lo12:DVPe]
	adrp	x4, .L.str.35
	add	x4, x4, :lo12:.L.str.35
	bl	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	movi	d0, #0000000000000000
	adrp	x22, DV10Mpe
	ldr	x0, [x13, :lo12:DV10Mpb]
	ldr	x1, [x22, :lo12:DV10Mpe]
	ldr	x2, [x5, :lo12:DV10pb]
	ldr	x3, [x6, :lo12:DV10pe]
	adrp	x4, .L.str.36
	add	x4, x4, :lo12:.L.str.36
	bl	_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	movi	d0, #0000000000000000
	adrp	x23, DV10MPb
	adrp	x24, DV10MPe
	ldr	x0, [x23, :lo12:DV10MPb]
	ldr	x1, [x24, :lo12:DV10MPe]
	ldr	x2, [x7, :lo12:DV10Pb]
	ldr	x3, [x8, :lo12:DV10Pe]
	adrp	x4, .L.str.37
	add	x4, x4, :lo12:.L.str.37
	bl	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	ldr	w25, [x4, :lo12:iterations]
	movi	d0, #0000000000000000
	ldr	x26, [x15, :lo12:dMpe]
	ldr	x2, [x0, :lo12:dpb]
	ldr	x3, [x1, :lo12:dpe]
	lsl	w30, w25, #3
	ldr	x25, [x14, :lo12:dMpb]
	mov	x1, x26
	str	w30, [x4, :lo12:iterations]
	adrp	x4, .L.str.38
	add	x4, x4, :lo12:.L.str.38
	mov	x0, x25
	bl	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x17, :lo12:dMPb]
	ldr	x1, [x18, :lo12:dMPe]
	ldr	x2, [x12, :lo12:dPb]
	ldr	x3, [x11, :lo12:dPe]
	adrp	x4, .L.str.39
	add	x4, x4, :lo12:.L.str.39
	bl	_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x16, :lo12:DVMpb]
	ldr	x1, [x19, :lo12:DVMpe]
	ldr	x2, [x2, :lo12:DVpb]
	ldr	x3, [x3, :lo12:DVpe]
	adrp	x4, .L.str.40
	add	x4, x4, :lo12:.L.str.40
	bl	_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x20, :lo12:DVMPb]
	ldr	x1, [x21, :lo12:DVMPe]
	ldr	x2, [x9, :lo12:DVPb]
	ldr	x3, [x10, :lo12:DVPe]
	adrp	x4, .L.str.41
	add	x4, x4, :lo12:.L.str.41
	bl	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x13, :lo12:DV10Mpb]
	ldr	x1, [x22, :lo12:DV10Mpe]
	ldr	x2, [x5, :lo12:DV10pb]
	ldr	x3, [x6, :lo12:DV10pe]
	adrp	x4, .L.str.42
	add	x4, x4, :lo12:.L.str.42
	bl	_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x23, :lo12:DV10MPb]
	ldr	x1, [x24, :lo12:DV10MPe]
	ldr	x2, [x7, :lo12:DV10Pb]
	ldr	x3, [x8, :lo12:DV10Pe]
	adrp	x4, .L.str.43
	add	x4, x4, :lo12:.L.str.43
	bl	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	movi	d0, #0000000000000000
	ldr	x14, [x14, :lo12:dMpb]
	ldr	x4, [x15, :lo12:dMpe]
	ldr	x0, [x0, :lo12:dpb]
	ldr	x1, [x1, :lo12:dpe]
	adrp	x15, .L.str.44
	add	x15, x15, :lo12:.L.str.44
	mov	x0, x14
	mov	x1, x4
	mov	x2, x14
	mov	x3, x4
	mov	x4, x15
	bl	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	movi	d0, #0000000000000000
	ldr	x1, [x17, :lo12:dMPb]
	ldr	x0, [x18, :lo12:dMPe]
	ldr	x2, [x12, :lo12:dPb]
	ldr	x3, [x11, :lo12:dPe]
	adrp	x4, .L.str.45
	add	x4, x4, :lo12:.L.str.45
	mov	x0, x1
	bl	_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	movi	d0, #0000000000000000
	ldr	x1, [x16, :lo12:DVMpb]
	ldr	x0, [x19, :lo12:DVMpe]
	ldr	x2, [x2, :lo12:DVpb]
	ldr	x3, [x3, :lo12:DVpe]
	adrp	x4, .L.str.46
	add	x4, x4, :lo12:.L.str.46
	mov	x0, x1
	bl	_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	movi	d0, #0000000000000000
	ldr	x1, [x20, :lo12:DVMPb]
	ldr	x0, [x21, :lo12:DVMPe]
	ldr	x2, [x9, :lo12:DVPb]
	ldr	x3, [x10, :lo12:DVPe]
	adrp	x4, .L.str.47
	add	x4, x4, :lo12:.L.str.47
	mov	x0, x1
	bl	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	movi	d0, #0000000000000000
	ldr	x1, [x13, :lo12:DV10Mpb]
	ldr	x0, [x22, :lo12:DV10Mpe]
	ldr	x2, [x5, :lo12:DV10pb]
	ldr	x3, [x6, :lo12:DV10pe]
	adrp	x4, .L.str.48
	add	x4, x4, :lo12:.L.str.48
	mov	x0, x1
	bl	_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	movi	d0, #0000000000000000
	ldr	x0, [x23, :lo12:DV10MPb]
	ldr	x1, [x24, :lo12:DV10MPe]
	ldr	x2, [x7, :lo12:DV10Pb]
	ldr	x3, [x8, :lo12:DV10Pe]
	adrp	x4, .L.str.49
	add	x4, x4, :lo12:.L.str.49
	bl	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	mov	w0, wzr
	.cfi_def_cfa wsp, 96
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #48]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #32]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #96             // 16-byte Folded Reload
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
	ret
.Lfunc_end5:
	.size	main, .Lfunc_end5-main
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc: // @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB6_51
// %bb.1:
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x20, x19, [sp, #16]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB6_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB6_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB6_6
.LBB6_4:                                //   in Loop: Header=BB6_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB6_5:                                //   in Loop: Header=BB6_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB6_50
.LBB6_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_8 Depth 2
                                        //     Child Loop BB6_10 Depth 2
                                        //     Child Loop BB6_13 Depth 2
                                        //       Child Loop BB6_14 Depth 3
                                        //     Child Loop BB6_18 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB6_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB6_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB6_8:                                //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB6_8
// %bb.9:                               //   in Loop: Header=BB6_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB6_11
.LBB6_10:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x18], #8
	cmp	x18, x5
	str	d0, [x17], #8
	b.ne	.LBB6_10
.LBB6_11:                               // %.preheader14
                                        //   in Loop: Header=BB6_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB6_13
.LBB6_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB6_13 Depth=2
	add	x19, x20, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	d0, [x19]
	cmp	x18, x0
	b.eq	.LBB6_17
.LBB6_13:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_14 Depth 3
	ldr	d0, [x18]
	mov	x19, x17
.LBB6_14:                               //   Parent Loop BB6_6 Depth=1
                                        //     Parent Loop BB6_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x20, x4, x19
	ldr	d1, [x20]
	fcmp	d0, d1
	b.pl	.LBB6_12
// %bb.15:                              //   in Loop: Header=BB6_14 Depth=3
	sub	x19, x19, #8
	str	d1, [x20, #8]
	cmn	x19, #8
	b.ne	.LBB6_14
// %bb.16:                              //   in Loop: Header=BB6_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	d0, [x4]
	cmp	x18, x0
	b.ne	.LBB6_13
.LBB6_17:                               // %.preheader12
                                        //   in Loop: Header=BB6_6 Depth=1
	mov	x17, x2
.LBB6_18:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB6_5
// %bb.19:                              //   in Loop: Header=BB6_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB6_18
	b	.LBB6_4
.LBB6_20:
	cmp	x6, x5
	b.eq	.LBB6_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB6_24
.LBB6_22:                               //   in Loop: Header=BB6_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB6_23:                               //   in Loop: Header=BB6_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB6_50
.LBB6_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_26 Depth 2
                                        //     Child Loop BB6_28 Depth 2
                                        //     Child Loop BB6_30 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB6_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB6_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB6_26:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB6_26
// %bb.27:                              //   in Loop: Header=BB6_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB6_29
.LBB6_28:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x18], #8
	cmp	x18, x5
	str	d0, [x17], #8
	b.ne	.LBB6_28
.LBB6_29:                               // %.preheader2
                                        //   in Loop: Header=BB6_24 Depth=1
	mov	x17, x2
.LBB6_30:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB6_23
// %bb.31:                              //   in Loop: Header=BB6_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB6_30
	b	.LBB6_22
.LBB6_32:                               // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB6_35
.LBB6_33:                               //   in Loop: Header=BB6_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB6_34:                               //   in Loop: Header=BB6_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB6_50
.LBB6_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_37 Depth 2
                                        //       Child Loop BB6_38 Depth 3
                                        //     Child Loop BB6_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB6_37
.LBB6_36:                               // %._crit_edge
                                        //   in Loop: Header=BB6_37 Depth=2
	add	x10, x11, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x10]
	cmp	x9, x0
	b.eq	.LBB6_41
.LBB6_37:                               //   Parent Loop BB6_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_38 Depth 3
	ldr	d0, [x9]
	mov	x10, x8
.LBB6_38:                               //   Parent Loop BB6_35 Depth=1
                                        //     Parent Loop BB6_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x11, x4, x10
	ldr	d1, [x11]
	fcmp	d0, d1
	b.pl	.LBB6_36
// %bb.39:                              //   in Loop: Header=BB6_38 Depth=3
	sub	x10, x10, #8
	str	d1, [x11, #8]
	cmn	x10, #8
	b.ne	.LBB6_38
// %bb.40:                              //   in Loop: Header=BB6_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x4]
	cmp	x9, x0
	b.ne	.LBB6_37
.LBB6_41:                               // %.preheader8
                                        //   in Loop: Header=BB6_35 Depth=1
	mov	x8, x2
.LBB6_42:                               //   Parent Loop BB6_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB6_34
// %bb.43:                              //   in Loop: Header=BB6_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB6_42
	b	.LBB6_33
.LBB6_44:                               // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB6_47
.LBB6_45:                               //   in Loop: Header=BB6_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB6_46:                               //   in Loop: Header=BB6_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB6_50
.LBB6_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_48 Depth 2
	mov	x7, x2
.LBB6_48:                               //   Parent Loop BB6_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB6_46
// %bb.49:                              //   in Loop: Header=BB6_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB6_48
	b	.LBB6_45
.LBB6_50:
	.cfi_def_cfa wsp, 32
	ldp	x20, x19, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB6_51:
	ret
.Lfunc_end6:
	.size	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end6-_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,"axG",@progbits,_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,comdat
	.weak	_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc // -- Begin function _Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,@function
_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc: // @_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB7_51
// %bb.1:
	stp	x29, x30, [sp, #-32]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	stp	x20, x19, [sp, #16]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB7_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB7_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB7_6
.LBB7_4:                                //   in Loop: Header=BB7_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB7_5:                                //   in Loop: Header=BB7_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB7_50
.LBB7_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_8 Depth 2
                                        //     Child Loop BB7_10 Depth 2
                                        //     Child Loop BB7_13 Depth 2
                                        //       Child Loop BB7_14 Depth 3
                                        //     Child Loop BB7_18 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB7_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB7_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB7_8:                                //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB7_8
// %bb.9:                               //   in Loop: Header=BB7_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB7_11
.LBB7_10:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x17], #8
	cmp	x17, x5
	str	d0, [x18], #8
	b.ne	.LBB7_10
.LBB7_11:                               // %.preheader14
                                        //   in Loop: Header=BB7_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB7_13
.LBB7_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB7_13 Depth=2
	add	x19, x20, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	d0, [x19]
	cmp	x18, x0
	b.eq	.LBB7_17
.LBB7_13:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB7_14 Depth 3
	ldr	d0, [x18]
	mov	x19, x17
.LBB7_14:                               //   Parent Loop BB7_6 Depth=1
                                        //     Parent Loop BB7_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x20, x4, x19
	ldr	d1, [x20]
	fcmp	d0, d1
	b.pl	.LBB7_12
// %bb.15:                              //   in Loop: Header=BB7_14 Depth=3
	sub	x19, x19, #8
	str	d1, [x20, #8]
	cmn	x19, #8
	b.ne	.LBB7_14
// %bb.16:                              //   in Loop: Header=BB7_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	d0, [x4]
	cmp	x18, x0
	b.ne	.LBB7_13
.LBB7_17:                               // %.preheader12
                                        //   in Loop: Header=BB7_6 Depth=1
	mov	x17, x2
.LBB7_18:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB7_5
// %bb.19:                              //   in Loop: Header=BB7_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB7_18
	b	.LBB7_4
.LBB7_20:
	cmp	x6, x5
	b.eq	.LBB7_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB7_24
.LBB7_22:                               //   in Loop: Header=BB7_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB7_23:                               //   in Loop: Header=BB7_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB7_50
.LBB7_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_26 Depth 2
                                        //     Child Loop BB7_28 Depth 2
                                        //     Child Loop BB7_30 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB7_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB7_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB7_26:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB7_26
// %bb.27:                              //   in Loop: Header=BB7_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB7_29
.LBB7_28:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x17], #8
	cmp	x17, x5
	str	d0, [x18], #8
	b.ne	.LBB7_28
.LBB7_29:                               // %.preheader2
                                        //   in Loop: Header=BB7_24 Depth=1
	mov	x17, x2
.LBB7_30:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB7_23
// %bb.31:                              //   in Loop: Header=BB7_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB7_30
	b	.LBB7_22
.LBB7_32:                               // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB7_35
.LBB7_33:                               //   in Loop: Header=BB7_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB7_34:                               //   in Loop: Header=BB7_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB7_50
.LBB7_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_37 Depth 2
                                        //       Child Loop BB7_38 Depth 3
                                        //     Child Loop BB7_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB7_37
.LBB7_36:                               // %._crit_edge
                                        //   in Loop: Header=BB7_37 Depth=2
	add	x10, x11, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x10]
	cmp	x9, x0
	b.eq	.LBB7_41
.LBB7_37:                               //   Parent Loop BB7_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB7_38 Depth 3
	ldr	d0, [x9]
	mov	x10, x8
.LBB7_38:                               //   Parent Loop BB7_35 Depth=1
                                        //     Parent Loop BB7_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x11, x4, x10
	ldr	d1, [x11]
	fcmp	d0, d1
	b.pl	.LBB7_36
// %bb.39:                              //   in Loop: Header=BB7_38 Depth=3
	sub	x10, x10, #8
	str	d1, [x11, #8]
	cmn	x10, #8
	b.ne	.LBB7_38
// %bb.40:                              //   in Loop: Header=BB7_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x4]
	cmp	x9, x0
	b.ne	.LBB7_37
.LBB7_41:                               // %.preheader8
                                        //   in Loop: Header=BB7_35 Depth=1
	mov	x8, x2
.LBB7_42:                               //   Parent Loop BB7_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB7_34
// %bb.43:                              //   in Loop: Header=BB7_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB7_42
	b	.LBB7_33
.LBB7_44:                               // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB7_47
.LBB7_45:                               //   in Loop: Header=BB7_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB7_46:                               //   in Loop: Header=BB7_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB7_50
.LBB7_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_48 Depth 2
	mov	x7, x2
.LBB7_48:                               //   Parent Loop BB7_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB7_46
// %bb.49:                              //   in Loop: Header=BB7_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB7_48
	b	.LBB7_45
.LBB7_50:
	.cfi_def_cfa wsp, 32
	ldp	x20, x19, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #32             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB7_51:
	ret
.Lfunc_end7:
	.size	_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc, .Lfunc_end7-_Z19test_insertion_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,"axG",@progbits,_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,comdat
	.weak	_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc // -- Begin function _Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,@function
_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc: // @_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB8_51
// %bb.1:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	str	x21, [sp, #16]                  // 8-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB8_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB8_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB8_6
.LBB8_4:                                //   in Loop: Header=BB8_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB8_5:                                //   in Loop: Header=BB8_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB8_50
.LBB8_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_8 Depth 2
                                        //     Child Loop BB8_10 Depth 2
                                        //     Child Loop BB8_13 Depth 2
                                        //       Child Loop BB8_14 Depth 3
                                        //     Child Loop BB8_18 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB8_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB8_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB8_8:                                //   Parent Loop BB8_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB8_8
// %bb.9:                               //   in Loop: Header=BB8_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB8_11
.LBB8_10:                               //   Parent Loop BB8_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x18], #8
	cmp	x18, x5
	str	x19, [x17], #8
	b.ne	.LBB8_10
.LBB8_11:                               // %.preheader14
                                        //   in Loop: Header=BB8_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB8_13
.LBB8_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB8_13 Depth=2
	add	x20, x21, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x20]
	cmp	x18, x0
	b.eq	.LBB8_17
.LBB8_13:                               //   Parent Loop BB8_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB8_14 Depth 3
	ldr	x19, [x18]
	mov	x20, x17
	fmov	d0, x19
.LBB8_14:                               //   Parent Loop BB8_6 Depth=1
                                        //     Parent Loop BB8_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x21, x4, x20
	ldr	d1, [x21]
	fcmp	d1, d0
	b.le	.LBB8_12
// %bb.15:                              //   in Loop: Header=BB8_14 Depth=3
	sub	x20, x20, #8
	str	d1, [x21, #8]
	cmn	x20, #8
	b.ne	.LBB8_14
// %bb.16:                              //   in Loop: Header=BB8_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x4]
	cmp	x18, x0
	b.ne	.LBB8_13
.LBB8_17:                               // %.preheader12
                                        //   in Loop: Header=BB8_6 Depth=1
	mov	x17, x2
.LBB8_18:                               //   Parent Loop BB8_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB8_5
// %bb.19:                              //   in Loop: Header=BB8_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB8_18
	b	.LBB8_4
.LBB8_20:
	cmp	x6, x5
	b.eq	.LBB8_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB8_24
.LBB8_22:                               //   in Loop: Header=BB8_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB8_23:                               //   in Loop: Header=BB8_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB8_50
.LBB8_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_26 Depth 2
                                        //     Child Loop BB8_28 Depth 2
                                        //     Child Loop BB8_30 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB8_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB8_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB8_26:                               //   Parent Loop BB8_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB8_26
// %bb.27:                              //   in Loop: Header=BB8_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB8_29
.LBB8_28:                               //   Parent Loop BB8_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x18], #8
	cmp	x18, x5
	str	x19, [x17], #8
	b.ne	.LBB8_28
.LBB8_29:                               // %.preheader2
                                        //   in Loop: Header=BB8_24 Depth=1
	mov	x17, x2
.LBB8_30:                               //   Parent Loop BB8_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB8_23
// %bb.31:                              //   in Loop: Header=BB8_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB8_30
	b	.LBB8_22
.LBB8_32:                               // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB8_35
.LBB8_33:                               //   in Loop: Header=BB8_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB8_34:                               //   in Loop: Header=BB8_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB8_50
.LBB8_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_37 Depth 2
                                        //       Child Loop BB8_38 Depth 3
                                        //     Child Loop BB8_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB8_37
.LBB8_36:                               // %._crit_edge
                                        //   in Loop: Header=BB8_37 Depth=2
	add	x11, x12, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x11]
	cmp	x9, x0
	b.eq	.LBB8_41
.LBB8_37:                               //   Parent Loop BB8_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB8_38 Depth 3
	ldr	x10, [x9]
	mov	x11, x8
	fmov	d0, x10
.LBB8_38:                               //   Parent Loop BB8_35 Depth=1
                                        //     Parent Loop BB8_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x4, x11
	ldr	d1, [x12]
	fcmp	d1, d0
	b.le	.LBB8_36
// %bb.39:                              //   in Loop: Header=BB8_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB8_38
// %bb.40:                              //   in Loop: Header=BB8_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x4]
	cmp	x9, x0
	b.ne	.LBB8_37
.LBB8_41:                               // %.preheader8
                                        //   in Loop: Header=BB8_35 Depth=1
	mov	x8, x2
.LBB8_42:                               //   Parent Loop BB8_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB8_34
// %bb.43:                              //   in Loop: Header=BB8_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB8_42
	b	.LBB8_33
.LBB8_44:                               // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB8_47
.LBB8_45:                               //   in Loop: Header=BB8_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB8_46:                               //   in Loop: Header=BB8_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB8_50
.LBB8_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_48 Depth 2
	mov	x7, x2
.LBB8_48:                               //   Parent Loop BB8_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB8_46
// %bb.49:                              //   in Loop: Header=BB8_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB8_48
	b	.LBB8_45
.LBB8_50:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x21, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w30
	.cfi_restore w29
.LBB8_51:
	ret
.Lfunc_end8:
	.size	_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc, .Lfunc_end8-_Z19test_insertion_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,"axG",@progbits,_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,comdat
	.weak	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc // -- Begin function _Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,@function
_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc: // @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB9_51
// %bb.1:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	str	x21, [sp, #16]                  // 8-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB9_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB9_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB9_6
.LBB9_4:                                //   in Loop: Header=BB9_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB9_5:                                //   in Loop: Header=BB9_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB9_50
.LBB9_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_8 Depth 2
                                        //     Child Loop BB9_10 Depth 2
                                        //     Child Loop BB9_13 Depth 2
                                        //       Child Loop BB9_14 Depth 3
                                        //     Child Loop BB9_18 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB9_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB9_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB9_8:                                //   Parent Loop BB9_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB9_8
// %bb.9:                               //   in Loop: Header=BB9_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB9_11
.LBB9_10:                               //   Parent Loop BB9_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x17], #8
	cmp	x17, x5
	str	x19, [x18], #8
	b.ne	.LBB9_10
.LBB9_11:                               // %.preheader14
                                        //   in Loop: Header=BB9_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB9_13
.LBB9_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB9_13 Depth=2
	add	x20, x21, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x20]
	cmp	x18, x0
	b.eq	.LBB9_17
.LBB9_13:                               //   Parent Loop BB9_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB9_14 Depth 3
	ldr	x19, [x18]
	mov	x20, x17
	fmov	d0, x19
.LBB9_14:                               //   Parent Loop BB9_6 Depth=1
                                        //     Parent Loop BB9_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x21, x4, x20
	ldr	d1, [x21]
	fcmp	d1, d0
	b.le	.LBB9_12
// %bb.15:                              //   in Loop: Header=BB9_14 Depth=3
	sub	x20, x20, #8
	str	d1, [x21, #8]
	cmn	x20, #8
	b.ne	.LBB9_14
// %bb.16:                              //   in Loop: Header=BB9_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x4]
	cmp	x18, x0
	b.ne	.LBB9_13
.LBB9_17:                               // %.preheader12
                                        //   in Loop: Header=BB9_6 Depth=1
	mov	x17, x2
.LBB9_18:                               //   Parent Loop BB9_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB9_5
// %bb.19:                              //   in Loop: Header=BB9_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB9_18
	b	.LBB9_4
.LBB9_20:
	cmp	x6, x5
	b.eq	.LBB9_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB9_24
.LBB9_22:                               //   in Loop: Header=BB9_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB9_23:                               //   in Loop: Header=BB9_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB9_50
.LBB9_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_26 Depth 2
                                        //     Child Loop BB9_28 Depth 2
                                        //     Child Loop BB9_30 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB9_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB9_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB9_26:                               //   Parent Loop BB9_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB9_26
// %bb.27:                              //   in Loop: Header=BB9_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB9_29
.LBB9_28:                               //   Parent Loop BB9_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x17], #8
	cmp	x17, x5
	str	x19, [x18], #8
	b.ne	.LBB9_28
.LBB9_29:                               // %.preheader2
                                        //   in Loop: Header=BB9_24 Depth=1
	mov	x17, x2
.LBB9_30:                               //   Parent Loop BB9_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB9_23
// %bb.31:                              //   in Loop: Header=BB9_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB9_30
	b	.LBB9_22
.LBB9_32:                               // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB9_35
.LBB9_33:                               //   in Loop: Header=BB9_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB9_34:                               //   in Loop: Header=BB9_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB9_50
.LBB9_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_37 Depth 2
                                        //       Child Loop BB9_38 Depth 3
                                        //     Child Loop BB9_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB9_37
.LBB9_36:                               // %._crit_edge
                                        //   in Loop: Header=BB9_37 Depth=2
	add	x11, x12, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x11]
	cmp	x9, x0
	b.eq	.LBB9_41
.LBB9_37:                               //   Parent Loop BB9_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB9_38 Depth 3
	ldr	x10, [x9]
	mov	x11, x8
	fmov	d0, x10
.LBB9_38:                               //   Parent Loop BB9_35 Depth=1
                                        //     Parent Loop BB9_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x4, x11
	ldr	d1, [x12]
	fcmp	d1, d0
	b.le	.LBB9_36
// %bb.39:                              //   in Loop: Header=BB9_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB9_38
// %bb.40:                              //   in Loop: Header=BB9_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x4]
	cmp	x9, x0
	b.ne	.LBB9_37
.LBB9_41:                               // %.preheader8
                                        //   in Loop: Header=BB9_35 Depth=1
	mov	x8, x2
.LBB9_42:                               //   Parent Loop BB9_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB9_34
// %bb.43:                              //   in Loop: Header=BB9_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB9_42
	b	.LBB9_33
.LBB9_44:                               // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB9_47
.LBB9_45:                               //   in Loop: Header=BB9_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB9_46:                               //   in Loop: Header=BB9_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB9_50
.LBB9_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_48 Depth 2
	mov	x7, x2
.LBB9_48:                               //   Parent Loop BB9_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB9_46
// %bb.49:                              //   in Loop: Header=BB9_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB9_48
	b	.LBB9_45
.LBB9_50:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x21, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w30
	.cfi_restore w29
.LBB9_51:
	ret
.Lfunc_end9:
	.size	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc, .Lfunc_end9-_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,"axG",@progbits,_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,comdat
	.weak	_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc // -- Begin function _Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,@function
_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc: // @_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB10_51
// %bb.1:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	str	x21, [sp, #16]                  // 8-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB10_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB10_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB10_6
.LBB10_4:                               //   in Loop: Header=BB10_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB10_5:                               //   in Loop: Header=BB10_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB10_50
.LBB10_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_8 Depth 2
                                        //     Child Loop BB10_10 Depth 2
                                        //     Child Loop BB10_13 Depth 2
                                        //       Child Loop BB10_14 Depth 3
                                        //     Child Loop BB10_18 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB10_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB10_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB10_8:                               //   Parent Loop BB10_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB10_8
// %bb.9:                               //   in Loop: Header=BB10_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB10_11
.LBB10_10:                              //   Parent Loop BB10_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x18], #8
	cmp	x18, x5
	str	x19, [x17], #8
	b.ne	.LBB10_10
.LBB10_11:                              // %.preheader14
                                        //   in Loop: Header=BB10_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB10_13
.LBB10_12:                              // %._crit_edge55
                                        //   in Loop: Header=BB10_13 Depth=2
	add	x20, x21, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x20]
	cmp	x18, x0
	b.eq	.LBB10_17
.LBB10_13:                              //   Parent Loop BB10_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_14 Depth 3
	ldr	x19, [x18]
	mov	x20, x17
	fmov	d0, x19
.LBB10_14:                              //   Parent Loop BB10_6 Depth=1
                                        //     Parent Loop BB10_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x21, x4, x20
	ldr	d1, [x21]
	fcmp	d1, d0
	b.le	.LBB10_12
// %bb.15:                              //   in Loop: Header=BB10_14 Depth=3
	sub	x20, x20, #8
	str	d1, [x21, #8]
	cmn	x20, #8
	b.ne	.LBB10_14
// %bb.16:                              //   in Loop: Header=BB10_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x4]
	cmp	x18, x0
	b.ne	.LBB10_13
.LBB10_17:                              // %.preheader12
                                        //   in Loop: Header=BB10_6 Depth=1
	mov	x17, x2
.LBB10_18:                              //   Parent Loop BB10_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB10_5
// %bb.19:                              //   in Loop: Header=BB10_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB10_18
	b	.LBB10_4
.LBB10_20:
	cmp	x6, x5
	b.eq	.LBB10_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x4, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x6, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x4, x14
	add	x14, x6, x14
	b	.LBB10_24
.LBB10_22:                              //   in Loop: Header=BB10_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB10_23:                              //   in Loop: Header=BB10_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB10_50
.LBB10_24:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_26 Depth 2
                                        //     Child Loop BB10_28 Depth 2
                                        //     Child Loop BB10_30 Depth 2
	mov	x17, x4
	mov	x18, x6
	tbnz	w12, #0, .LBB10_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB10_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB10_26:                              //   Parent Loop BB10_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x18, #-16]
	subs	x19, x19, #4
	add	x18, x18, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB10_26
// %bb.27:                              //   in Loop: Header=BB10_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB10_29
.LBB10_28:                              //   Parent Loop BB10_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x18], #8
	cmp	x18, x5
	str	x19, [x17], #8
	b.ne	.LBB10_28
.LBB10_29:                              // %.preheader2
                                        //   in Loop: Header=BB10_24 Depth=1
	mov	x17, x2
.LBB10_30:                              //   Parent Loop BB10_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB10_23
// %bb.31:                              //   in Loop: Header=BB10_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB10_30
	b	.LBB10_22
.LBB10_32:                              // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB10_35
.LBB10_33:                              //   in Loop: Header=BB10_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB10_34:                              //   in Loop: Header=BB10_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB10_50
.LBB10_35:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_37 Depth 2
                                        //       Child Loop BB10_38 Depth 3
                                        //     Child Loop BB10_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB10_37
.LBB10_36:                              // %._crit_edge
                                        //   in Loop: Header=BB10_37 Depth=2
	add	x11, x12, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x11]
	cmp	x9, x0
	b.eq	.LBB10_41
.LBB10_37:                              //   Parent Loop BB10_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB10_38 Depth 3
	ldr	x10, [x9]
	mov	x11, x8
	fmov	d0, x10
.LBB10_38:                              //   Parent Loop BB10_35 Depth=1
                                        //     Parent Loop BB10_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x4, x11
	ldr	d1, [x12]
	fcmp	d1, d0
	b.le	.LBB10_36
// %bb.39:                              //   in Loop: Header=BB10_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB10_38
// %bb.40:                              //   in Loop: Header=BB10_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x4]
	cmp	x9, x0
	b.ne	.LBB10_37
.LBB10_41:                              // %.preheader8
                                        //   in Loop: Header=BB10_35 Depth=1
	mov	x8, x2
.LBB10_42:                              //   Parent Loop BB10_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB10_34
// %bb.43:                              //   in Loop: Header=BB10_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_42
	b	.LBB10_33
.LBB10_44:                              // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB10_47
.LBB10_45:                              //   in Loop: Header=BB10_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB10_46:                              //   in Loop: Header=BB10_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB10_50
.LBB10_47:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_48 Depth 2
	mov	x7, x2
.LBB10_48:                              //   Parent Loop BB10_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB10_46
// %bb.49:                              //   in Loop: Header=BB10_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB10_48
	b	.LBB10_45
.LBB10_50:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x21, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w30
	.cfi_restore w29
.LBB10_51:
	ret
.Lfunc_end10:
	.size	_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc, .Lfunc_end10-_Z19test_insertion_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,"axG",@progbits,_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,comdat
	.weak	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc // -- Begin function _Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,@function
_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc: // @_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x5, x1
	adrp	x1, iterations
	mov	x0, x3
	ldr	w3, [x1, :lo12:iterations]
	cmp	w3, #1
	b.lt	.LBB11_51
// %bb.1:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	str	x21, [sp, #16]                  // 8-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	x4, x2
	add	x2, x2, #8
	mov	x6, x0
	cmp	x2, x0
	b.eq	.LBB11_20
// %bb.2:
	cmp	x6, x5
	b.eq	.LBB11_32
// %bb.3:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB11_6
.LBB11_4:                               //   in Loop: Header=BB11_6 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB11_5:                               //   in Loop: Header=BB11_6 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB11_50
.LBB11_6:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_8 Depth 2
                                        //     Child Loop BB11_10 Depth 2
                                        //     Child Loop BB11_13 Depth 2
                                        //       Child Loop BB11_14 Depth 3
                                        //     Child Loop BB11_18 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB11_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB11_6 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB11_8:                               //   Parent Loop BB11_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB11_8
// %bb.9:                               //   in Loop: Header=BB11_6 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB11_11
.LBB11_10:                              //   Parent Loop BB11_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x17], #8
	cmp	x17, x5
	str	x19, [x18], #8
	b.ne	.LBB11_10
.LBB11_11:                              // %.preheader14
                                        //   in Loop: Header=BB11_6 Depth=1
	mov	x17, xzr
	mov	x18, x2
	b	.LBB11_13
.LBB11_12:                              // %._crit_edge55
                                        //   in Loop: Header=BB11_13 Depth=2
	add	x20, x21, #8
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x20]
	cmp	x18, x0
	b.eq	.LBB11_17
.LBB11_13:                              //   Parent Loop BB11_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB11_14 Depth 3
	ldr	x19, [x18]
	mov	x20, x17
	fmov	d0, x19
.LBB11_14:                              //   Parent Loop BB11_6 Depth=1
                                        //     Parent Loop BB11_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x21, x4, x20
	ldr	d1, [x21]
	fcmp	d1, d0
	b.le	.LBB11_12
// %bb.15:                              //   in Loop: Header=BB11_14 Depth=3
	sub	x20, x20, #8
	str	d1, [x21, #8]
	cmn	x20, #8
	b.ne	.LBB11_14
// %bb.16:                              //   in Loop: Header=BB11_13 Depth=2
	add	x18, x18, #8
	add	x17, x17, #8
	str	x19, [x4]
	cmp	x18, x0
	b.ne	.LBB11_13
.LBB11_17:                              // %.preheader12
                                        //   in Loop: Header=BB11_6 Depth=1
	mov	x17, x2
.LBB11_18:                              //   Parent Loop BB11_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB11_5
// %bb.19:                              //   in Loop: Header=BB11_18 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB11_18
	b	.LBB11_4
.LBB11_20:
	cmp	x6, x5
	b.eq	.LBB11_44
// %bb.21:
	sub	x8, x5, x6
	sub	x12, x4, x6
	mov	w13, #32                        // =0x20
	sub	x10, x8, #8
	mov	w7, wzr
	add	x11, x6, #16
	lsr	x8, x10, #3
	cmp	x10, #24
	add	x10, x4, #16
	ccmp	x12, x13, #0, hs
	adrp	x15, current_test
	adrp	x16, .L.str.51
	add	x16, x16, :lo12:.L.str.51
	add	x8, x8, #1
	cset	w12, lo
	and	x9, x8, #0x3ffffffffffffffc
	lsl	x14, x9, #3
	add	x13, x6, x14
	add	x14, x4, x14
	b	.LBB11_24
.LBB11_22:                              //   in Loop: Header=BB11_24 Depth=1
	ldr	w1, [x15, :lo12:current_test]
	mov	x0, x16
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB11_23:                              //   in Loop: Header=BB11_24 Depth=1
	add	w7, w7, #1
	cmp	w7, w3
	b.ge	.LBB11_50
.LBB11_24:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_26 Depth 2
                                        //     Child Loop BB11_28 Depth 2
                                        //     Child Loop BB11_30 Depth 2
	mov	x17, x6
	mov	x18, x4
	tbnz	w12, #0, .LBB11_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB11_24 Depth=1
	mov	x17, x11
	mov	x18, x10
	mov	x19, x9
.LBB11_26:                              //   Parent Loop BB11_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x19, x19, #4
	add	x17, x17, #32
	stp	q0, q1, [x18, #-16]
	add	x18, x18, #32
	b.ne	.LBB11_26
// %bb.27:                              //   in Loop: Header=BB11_24 Depth=1
	cmp	x8, x9
	mov	x17, x13
	mov	x18, x14
	b.eq	.LBB11_29
.LBB11_28:                              //   Parent Loop BB11_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x19, [x17], #8
	cmp	x17, x5
	str	x19, [x18], #8
	b.ne	.LBB11_28
.LBB11_29:                              // %.preheader2
                                        //   in Loop: Header=BB11_24 Depth=1
	mov	x17, x2
.LBB11_30:                              //   Parent Loop BB11_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x17, x0
	b.eq	.LBB11_23
// %bb.31:                              //   in Loop: Header=BB11_30 Depth=2
	ldp	d1, d0, [x17, #-8]
	add	x17, x17, #8
	fcmp	d0, d1
	b.pl	.LBB11_30
	b	.LBB11_22
.LBB11_32:                              // %.preheader10
	mov	w5, wzr
	adrp	x6, current_test
	adrp	x7, .L.str.51
	add	x7, x7, :lo12:.L.str.51
	b	.LBB11_35
.LBB11_33:                              //   in Loop: Header=BB11_35 Depth=1
	ldr	w1, [x6, :lo12:current_test]
	mov	x0, x7
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB11_34:                              //   in Loop: Header=BB11_35 Depth=1
	add	w5, w5, #1
	cmp	w5, w3
	b.ge	.LBB11_50
.LBB11_35:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_37 Depth 2
                                        //       Child Loop BB11_38 Depth 3
                                        //     Child Loop BB11_42 Depth 2
	mov	x8, xzr
	mov	x9, x2
	b	.LBB11_37
.LBB11_36:                              // %._crit_edge
                                        //   in Loop: Header=BB11_37 Depth=2
	add	x11, x12, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x11]
	cmp	x9, x0
	b.eq	.LBB11_41
.LBB11_37:                              //   Parent Loop BB11_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB11_38 Depth 3
	ldr	x10, [x9]
	mov	x11, x8
	fmov	d0, x10
.LBB11_38:                              //   Parent Loop BB11_35 Depth=1
                                        //     Parent Loop BB11_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x4, x11
	ldr	d1, [x12]
	fcmp	d1, d0
	b.le	.LBB11_36
// %bb.39:                              //   in Loop: Header=BB11_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB11_38
// %bb.40:                              //   in Loop: Header=BB11_37 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	x10, [x4]
	cmp	x9, x0
	b.ne	.LBB11_37
.LBB11_41:                              // %.preheader8
                                        //   in Loop: Header=BB11_35 Depth=1
	mov	x8, x2
.LBB11_42:                              //   Parent Loop BB11_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x0
	b.eq	.LBB11_34
// %bb.43:                              //   in Loop: Header=BB11_42 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB11_42
	b	.LBB11_33
.LBB11_44:                              // %.preheader
	mov	w4, wzr
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB11_47
.LBB11_45:                              //   in Loop: Header=BB11_47 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
	ldr	w3, [x1, :lo12:iterations]
.LBB11_46:                              //   in Loop: Header=BB11_47 Depth=1
	add	w4, w4, #1
	cmp	w4, w3
	b.ge	.LBB11_50
.LBB11_47:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_48 Depth 2
	mov	x7, x2
.LBB11_48:                              //   Parent Loop BB11_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB11_46
// %bb.49:                              //   in Loop: Header=BB11_48 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB11_48
	b	.LBB11_45
.LBB11_50:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldr	x21, [sp, #16]                  // 8-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w30
	.cfi_restore w29
.LBB11_51:
	ret
.Lfunc_end11:
	.size	_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc, .Lfunc_end11-_Z19test_insertion_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc: // @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB12_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB12_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB12_5
.LBB12_3:                               //   in Loop: Header=BB12_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB12_4:                               //   in Loop: Header=BB12_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB12_19
.LBB12_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB12_7 Depth 2
                                        //     Child Loop BB12_9 Depth 2
                                        //     Child Loop BB12_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB12_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB12_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB12_7:                               //   Parent Loop BB12_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB12_7
// %bb.8:                               //   in Loop: Header=BB12_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB12_10
.LBB12_9:                               //   Parent Loop BB12_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x17], #8
	cmp	x17, x3
	str	d0, [x16], #8
	b.ne	.LBB12_9
.LBB12_10:                              //   in Loop: Header=BB12_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
	mov	x16, x10
.LBB12_11:                              //   Parent Loop BB12_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB12_4
// %bb.12:                              //   in Loop: Header=BB12_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB12_11
	b	.LBB12_3
.LBB12_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB12_16
.LBB12_14:                              //   in Loop: Header=BB12_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB12_15:                              //   in Loop: Header=BB12_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB12_19
.LBB12_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB12_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
	mov	x7, x4
.LBB12_17:                              //   Parent Loop BB12_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB12_15
// %bb.18:                              //   in Loop: Header=BB12_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB12_17
	b	.LBB12_14
.LBB12_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB12_20:
	ret
.Lfunc_end12:
	.size	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end12-_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,"axG",@progbits,_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,comdat
	.weak	_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc // -- Begin function _Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,@function
_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc: // @_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB13_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB13_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB13_5
.LBB13_3:                               //   in Loop: Header=BB13_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB13_4:                               //   in Loop: Header=BB13_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB13_19
.LBB13_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_7 Depth 2
                                        //     Child Loop BB13_9 Depth 2
                                        //     Child Loop BB13_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB13_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB13_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB13_7:                               //   Parent Loop BB13_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB13_7
// %bb.8:                               //   in Loop: Header=BB13_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB13_10
.LBB13_9:                               //   Parent Loop BB13_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x16], #8
	cmp	x16, x3
	str	d0, [x17], #8
	b.ne	.LBB13_9
.LBB13_10:                              //   in Loop: Header=BB13_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	mov	x16, x10
.LBB13_11:                              //   Parent Loop BB13_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB13_4
// %bb.12:                              //   in Loop: Header=BB13_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB13_11
	b	.LBB13_3
.LBB13_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB13_16
.LBB13_14:                              //   in Loop: Header=BB13_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB13_15:                              //   in Loop: Header=BB13_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB13_19
.LBB13_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	mov	x7, x4
.LBB13_17:                              //   Parent Loop BB13_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB13_15
// %bb.18:                              //   in Loop: Header=BB13_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB13_17
	b	.LBB13_14
.LBB13_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB13_20:
	ret
.Lfunc_end13:
	.size	_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc, .Lfunc_end13-_Z14test_quicksortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,"axG",@progbits,_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,comdat
	.weak	_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc // -- Begin function _Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,@function
_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc: // @_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB14_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB14_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB14_5
.LBB14_3:                               //   in Loop: Header=BB14_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB14_4:                               //   in Loop: Header=BB14_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB14_19
.LBB14_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_7 Depth 2
                                        //     Child Loop BB14_9 Depth 2
                                        //     Child Loop BB14_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB14_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB14_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB14_7:                               //   Parent Loop BB14_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB14_7
// %bb.8:                               //   in Loop: Header=BB14_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB14_10
.LBB14_9:                               //   Parent Loop BB14_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x17], #8
	cmp	x17, x3
	str	x18, [x16], #8
	b.ne	.LBB14_9
.LBB14_10:                              //   in Loop: Header=BB14_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	mov	x16, x10
.LBB14_11:                              //   Parent Loop BB14_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB14_4
// %bb.12:                              //   in Loop: Header=BB14_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB14_11
	b	.LBB14_3
.LBB14_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB14_16
.LBB14_14:                              //   in Loop: Header=BB14_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB14_15:                              //   in Loop: Header=BB14_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB14_19
.LBB14_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	mov	x7, x4
.LBB14_17:                              //   Parent Loop BB14_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB14_15
// %bb.18:                              //   in Loop: Header=BB14_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB14_17
	b	.LBB14_14
.LBB14_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB14_20:
	ret
.Lfunc_end14:
	.size	_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc, .Lfunc_end14-_Z14test_quicksortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,"axG",@progbits,_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,comdat
	.weak	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc // -- Begin function _Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,@function
_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc: // @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB15_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB15_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB15_5
.LBB15_3:                               //   in Loop: Header=BB15_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB15_4:                               //   in Loop: Header=BB15_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB15_19
.LBB15_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_7 Depth 2
                                        //     Child Loop BB15_9 Depth 2
                                        //     Child Loop BB15_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB15_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB15_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB15_7:                               //   Parent Loop BB15_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB15_7
// %bb.8:                               //   in Loop: Header=BB15_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB15_10
.LBB15_9:                               //   Parent Loop BB15_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x16], #8
	cmp	x16, x3
	str	x18, [x17], #8
	b.ne	.LBB15_9
.LBB15_10:                              //   in Loop: Header=BB15_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	mov	x16, x10
.LBB15_11:                              //   Parent Loop BB15_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB15_4
// %bb.12:                              //   in Loop: Header=BB15_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB15_11
	b	.LBB15_3
.LBB15_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB15_16
.LBB15_14:                              //   in Loop: Header=BB15_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB15_15:                              //   in Loop: Header=BB15_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB15_19
.LBB15_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	mov	x7, x4
.LBB15_17:                              //   Parent Loop BB15_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB15_15
// %bb.18:                              //   in Loop: Header=BB15_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB15_17
	b	.LBB15_14
.LBB15_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB15_20:
	ret
.Lfunc_end15:
	.size	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc, .Lfunc_end15-_Z14test_quicksortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,"axG",@progbits,_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,comdat
	.weak	_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc // -- Begin function _Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,@function
_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc: // @_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB16_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB16_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB16_5
.LBB16_3:                               //   in Loop: Header=BB16_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB16_4:                               //   in Loop: Header=BB16_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB16_19
.LBB16_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_7 Depth 2
                                        //     Child Loop BB16_9 Depth 2
                                        //     Child Loop BB16_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB16_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB16_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB16_7:                               //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB16_7
// %bb.8:                               //   in Loop: Header=BB16_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB16_10
.LBB16_9:                               //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x17], #8
	cmp	x17, x3
	str	x18, [x16], #8
	b.ne	.LBB16_9
.LBB16_10:                              //   in Loop: Header=BB16_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	mov	x16, x10
.LBB16_11:                              //   Parent Loop BB16_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB16_4
// %bb.12:                              //   in Loop: Header=BB16_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB16_11
	b	.LBB16_3
.LBB16_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB16_16
.LBB16_14:                              //   in Loop: Header=BB16_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB16_15:                              //   in Loop: Header=BB16_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB16_19
.LBB16_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	mov	x7, x4
.LBB16_17:                              //   Parent Loop BB16_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB16_15
// %bb.18:                              //   in Loop: Header=BB16_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB16_17
	b	.LBB16_14
.LBB16_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB16_20:
	ret
.Lfunc_end16:
	.size	_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc, .Lfunc_end16-_Z14test_quicksortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,"axG",@progbits,_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,comdat
	.weak	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc // -- Begin function _Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,@function
_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc: // @_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB17_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB17_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB17_5
.LBB17_3:                               //   in Loop: Header=BB17_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB17_4:                               //   in Loop: Header=BB17_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB17_19
.LBB17_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_7 Depth 2
                                        //     Child Loop BB17_9 Depth 2
                                        //     Child Loop BB17_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB17_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB17_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB17_7:                               //   Parent Loop BB17_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB17_7
// %bb.8:                               //   in Loop: Header=BB17_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB17_10
.LBB17_9:                               //   Parent Loop BB17_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x16], #8
	cmp	x16, x3
	str	x18, [x17], #8
	b.ne	.LBB17_9
.LBB17_10:                              //   in Loop: Header=BB17_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	mov	x16, x10
.LBB17_11:                              //   Parent Loop BB17_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB17_4
// %bb.12:                              //   in Loop: Header=BB17_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB17_11
	b	.LBB17_3
.LBB17_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB17_16
.LBB17_14:                              //   in Loop: Header=BB17_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB17_15:                              //   in Loop: Header=BB17_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB17_19
.LBB17_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	mov	x7, x4
.LBB17_17:                              //   Parent Loop BB17_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB17_15
// %bb.18:                              //   in Loop: Header=BB17_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB17_17
	b	.LBB17_14
.LBB17_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB17_20:
	ret
.Lfunc_end17:
	.size	_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc, .Lfunc_end17-_Z14test_quicksortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc: // @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB18_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB18_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB18_5
.LBB18_3:                               //   in Loop: Header=BB18_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB18_4:                               //   in Loop: Header=BB18_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB18_19
.LBB18_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_7 Depth 2
                                        //     Child Loop BB18_9 Depth 2
                                        //     Child Loop BB18_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB18_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB18_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB18_7:                               //   Parent Loop BB18_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB18_7
// %bb.8:                               //   in Loop: Header=BB18_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB18_10
.LBB18_9:                               //   Parent Loop BB18_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x17], #8
	cmp	x17, x3
	str	d0, [x16], #8
	b.ne	.LBB18_9
.LBB18_10:                              //   in Loop: Header=BB18_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIPddEEvT_S2_
	mov	x16, x10
.LBB18_11:                              //   Parent Loop BB18_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB18_4
// %bb.12:                              //   in Loop: Header=BB18_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB18_11
	b	.LBB18_3
.LBB18_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB18_16
.LBB18_14:                              //   in Loop: Header=BB18_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB18_15:                              //   in Loop: Header=BB18_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB18_19
.LBB18_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIPddEEvT_S2_
	mov	x7, x4
.LBB18_17:                              //   Parent Loop BB18_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB18_15
// %bb.18:                              //   in Loop: Header=BB18_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB18_17
	b	.LBB18_14
.LBB18_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB18_20:
	ret
.Lfunc_end18:
	.size	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end18-_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,"axG",@progbits,_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,comdat
	.weak	_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc // -- Begin function _Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc,@function
_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc: // @_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB19_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB19_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB19_5
.LBB19_3:                               //   in Loop: Header=BB19_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB19_4:                               //   in Loop: Header=BB19_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB19_19
.LBB19_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_7 Depth 2
                                        //     Child Loop BB19_9 Depth 2
                                        //     Child Loop BB19_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB19_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB19_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB19_7:                               //   Parent Loop BB19_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB19_7
// %bb.8:                               //   in Loop: Header=BB19_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB19_10
.LBB19_9:                               //   Parent Loop BB19_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x16], #8
	cmp	x16, x3
	str	d0, [x17], #8
	b.ne	.LBB19_9
.LBB19_10:                              //   in Loop: Header=BB19_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_
	mov	x16, x10
.LBB19_11:                              //   Parent Loop BB19_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB19_4
// %bb.12:                              //   in Loop: Header=BB19_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB19_11
	b	.LBB19_3
.LBB19_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB19_16
.LBB19_14:                              //   in Loop: Header=BB19_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB19_15:                              //   in Loop: Header=BB19_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB19_19
.LBB19_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_
	mov	x7, x4
.LBB19_17:                              //   Parent Loop BB19_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB19_15
// %bb.18:                              //   in Loop: Header=BB19_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB19_17
	b	.LBB19_14
.LBB19_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB19_20:
	ret
.Lfunc_end19:
	.size	_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc, .Lfunc_end19-_Z14test_heap_sortI14PointerWrapperIdEdEvT_S2_S2_S2_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,"axG",@progbits,_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,comdat
	.weak	_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc // -- Begin function _Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc,@function
_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc: // @_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB20_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB20_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB20_5
.LBB20_3:                               //   in Loop: Header=BB20_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB20_4:                               //   in Loop: Header=BB20_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB20_19
.LBB20_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_7 Depth 2
                                        //     Child Loop BB20_9 Depth 2
                                        //     Child Loop BB20_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB20_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB20_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB20_7:                               //   Parent Loop BB20_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB20_7
// %bb.8:                               //   in Loop: Header=BB20_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB20_10
.LBB20_9:                               //   Parent Loop BB20_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x17], #8
	cmp	x17, x3
	str	x18, [x16], #8
	b.ne	.LBB20_9
.LBB20_10:                              //   in Loop: Header=BB20_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_
	mov	x16, x10
.LBB20_11:                              //   Parent Loop BB20_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB20_4
// %bb.12:                              //   in Loop: Header=BB20_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB20_11
	b	.LBB20_3
.LBB20_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB20_16
.LBB20_14:                              //   in Loop: Header=BB20_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB20_15:                              //   in Loop: Header=BB20_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB20_19
.LBB20_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_
	mov	x7, x4
.LBB20_17:                              //   Parent Loop BB20_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB20_15
// %bb.18:                              //   in Loop: Header=BB20_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB20_17
	b	.LBB20_14
.LBB20_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB20_20:
	ret
.Lfunc_end20:
	.size	_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc, .Lfunc_end20-_Z14test_heap_sortIP12ValueWrapperIdES1_EvT_S3_S3_S3_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,"axG",@progbits,_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,comdat
	.weak	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc // -- Begin function _Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc,@function
_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc: // @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB21_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB21_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB21_5
.LBB21_3:                               //   in Loop: Header=BB21_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB21_4:                               //   in Loop: Header=BB21_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB21_19
.LBB21_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_7 Depth 2
                                        //     Child Loop BB21_9 Depth 2
                                        //     Child Loop BB21_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB21_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB21_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB21_7:                               //   Parent Loop BB21_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB21_7
// %bb.8:                               //   in Loop: Header=BB21_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB21_10
.LBB21_9:                               //   Parent Loop BB21_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x16], #8
	cmp	x16, x3
	str	x18, [x17], #8
	b.ne	.LBB21_9
.LBB21_10:                              //   in Loop: Header=BB21_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	mov	x16, x10
.LBB21_11:                              //   Parent Loop BB21_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB21_4
// %bb.12:                              //   in Loop: Header=BB21_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB21_11
	b	.LBB21_3
.LBB21_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB21_16
.LBB21_14:                              //   in Loop: Header=BB21_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB21_15:                              //   in Loop: Header=BB21_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB21_19
.LBB21_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	mov	x7, x4
.LBB21_17:                              //   Parent Loop BB21_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB21_15
// %bb.18:                              //   in Loop: Header=BB21_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB21_17
	b	.LBB21_14
.LBB21_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB21_20:
	ret
.Lfunc_end21:
	.size	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc, .Lfunc_end21-_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIdEES2_EvT_S4_S4_S4_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,"axG",@progbits,_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,comdat
	.weak	_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc // -- Begin function _Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc,@function
_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc: // @_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB22_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB22_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x4, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x1, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x1, x13
	add	x13, x4, x13
	b	.LBB22_5
.LBB22_3:                               //   in Loop: Header=BB22_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB22_4:                               //   in Loop: Header=BB22_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB22_19
.LBB22_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_7 Depth 2
                                        //     Child Loop BB22_9 Depth 2
                                        //     Child Loop BB22_11 Depth 2
	mov	x16, x1
	mov	x17, x4
	tbnz	w11, #0, .LBB22_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB22_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB22_7:                               //   Parent Loop BB22_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x17, #-16]
	subs	x18, x18, #4
	add	x17, x17, #32
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB22_7
// %bb.8:                               //   in Loop: Header=BB22_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB22_10
.LBB22_9:                               //   Parent Loop BB22_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x17], #8
	cmp	x17, x3
	str	x18, [x16], #8
	b.ne	.LBB22_9
.LBB22_10:                              //   in Loop: Header=BB22_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	mov	x16, x10
.LBB22_11:                              //   Parent Loop BB22_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB22_4
// %bb.12:                              //   in Loop: Header=BB22_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB22_11
	b	.LBB22_3
.LBB22_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB22_16
.LBB22_14:                              //   in Loop: Header=BB22_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB22_15:                              //   in Loop: Header=BB22_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB22_19
.LBB22_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	mov	x7, x4
.LBB22_17:                              //   Parent Loop BB22_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB22_15
// %bb.18:                              //   in Loop: Header=BB22_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB22_17
	b	.LBB22_14
.LBB22_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB22_20:
	ret
.Lfunc_end22:
	.size	_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc, .Lfunc_end22-_Z14test_heap_sortIP12ValueWrapperIS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IS0_IdEEEEEEEEEESA_EvT_SC_SC_SC_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,"axG",@progbits,_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,comdat
	.weak	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc // -- Begin function _Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc,@function
_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc: // @_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_startproc
// %bb.0:
	mov	x1, x2
	adrp	x2, iterations
	ldr	w4, [x2, :lo12:iterations]
	cmp	w4, #1
	b.lt	.LBB23_20
// %bb.1:
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x3
	mov	x3, x1
	mov	x4, x0
	cmp	x0, x1
	b.eq	.LBB23_13
// %bb.2:
	sub	x6, x3, x4
	sub	x11, x1, x4
	mov	w12, #32                        // =0x20
	sub	x9, x6, #8
	mov	w5, wzr
	add	x8, x1, #16
	lsr	x6, x9, #3
	cmp	x9, #24
	add	x9, x4, #16
	ccmp	x11, x12, #0, hs
	add	x10, x1, #8
	adrp	x14, current_test
	add	x6, x6, #1
	cset	w11, lo
	adrp	x15, .L.str.51
	add	x15, x15, :lo12:.L.str.51
	and	x7, x6, #0x3ffffffffffffffc
	lsl	x13, x7, #3
	add	x12, x4, x13
	add	x13, x1, x13
	b	.LBB23_5
.LBB23_3:                               //   in Loop: Header=BB23_5 Depth=1
	ldr	w1, [x14, :lo12:current_test]
	mov	x0, x15
	bl	printf
.LBB23_4:                               //   in Loop: Header=BB23_5 Depth=1
	ldr	w16, [x2, :lo12:iterations]
	add	w5, w5, #1
	cmp	w5, w16
	b.ge	.LBB23_19
.LBB23_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_7 Depth 2
                                        //     Child Loop BB23_9 Depth 2
                                        //     Child Loop BB23_11 Depth 2
	mov	x16, x4
	mov	x17, x1
	tbnz	w11, #0, .LBB23_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB23_5 Depth=1
	mov	x16, x9
	mov	x17, x8
	mov	x18, x7
.LBB23_7:                               //   Parent Loop BB23_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x16, #-16]
	subs	x18, x18, #4
	add	x16, x16, #32
	stp	q0, q1, [x17, #-16]
	add	x17, x17, #32
	b.ne	.LBB23_7
// %bb.8:                               //   in Loop: Header=BB23_5 Depth=1
	cmp	x6, x7
	mov	x16, x12
	mov	x17, x13
	b.eq	.LBB23_10
.LBB23_9:                               //   Parent Loop BB23_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	x18, [x16], #8
	cmp	x16, x3
	str	x18, [x17], #8
	b.ne	.LBB23_9
.LBB23_10:                              //   in Loop: Header=BB23_5 Depth=1
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	mov	x16, x10
.LBB23_11:                              //   Parent Loop BB23_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x16, x0
	b.eq	.LBB23_4
// %bb.12:                              //   in Loop: Header=BB23_11 Depth=2
	ldp	d1, d0, [x16, #-8]
	add	x16, x16, #8
	fcmp	d0, d1
	b.pl	.LBB23_11
	b	.LBB23_3
.LBB23_13:                              // %.preheader
	mov	w3, wzr
	add	x4, x1, #8
	adrp	x5, current_test
	adrp	x6, .L.str.51
	add	x6, x6, :lo12:.L.str.51
	b	.LBB23_16
.LBB23_14:                              //   in Loop: Header=BB23_16 Depth=1
	ldr	w1, [x5, :lo12:current_test]
	mov	x0, x6
	bl	printf
.LBB23_15:                              //   in Loop: Header=BB23_16 Depth=1
	ldr	w7, [x2, :lo12:iterations]
	add	w3, w3, #1
	cmp	w3, w7
	b.ge	.LBB23_19
.LBB23_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_17 Depth 2
	mov	x0, x1
	bl	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	mov	x7, x4
.LBB23_17:                              //   Parent Loop BB23_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x7, x0
	b.eq	.LBB23_15
// %bb.18:                              //   in Loop: Header=BB23_17 Depth=2
	ldp	d1, d0, [x7, #-8]
	add	x7, x7, #8
	fcmp	d0, d1
	b.pl	.LBB23_17
	b	.LBB23_14
.LBB23_19:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB23_20:
	ret
.Lfunc_end23:
	.size	_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc, .Lfunc_end23-_Z14test_heap_sortI14PointerWrapperI12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEEESB_EvT_SD_SD_SD_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortIPddEEvT_S2_,"axG",@progbits,_ZN9benchmark9quicksortIPddEEvT_S2_,comdat
	.weak	_ZN9benchmark9quicksortIPddEEvT_S2_ // -- Begin function _ZN9benchmark9quicksortIPddEEvT_S2_
	.p2align	2
	.type	_ZN9benchmark9quicksortIPddEEvT_S2_,@function
_ZN9benchmark9quicksortIPddEEvT_S2_:    // @_ZN9benchmark9quicksortIPddEEvT_S2_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB24_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB24_3
.LBB24_2:                               //   in Loop: Header=BB24_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB24_11
.LBB24_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB24_4 Depth 2
                                        //       Child Loop BB24_5 Depth 3
                                        //       Child Loop BB24_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB24_4:                               //   Parent Loop BB24_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB24_5 Depth 3
                                        //       Child Loop BB24_8 Depth 3
	sub	x5, x2, x3
.LBB24_5:                               //   Parent Loop BB24_3 Depth=1
                                        //     Parent Loop BB24_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB24_5
// %bb.6:                               //   in Loop: Header=BB24_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB24_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB24_4 Depth=2
	sub	x4, x4, #8
.LBB24_8:                               //   Parent Loop BB24_3 Depth=1
                                        //     Parent Loop BB24_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB24_8
// %bb.9:                               //   in Loop: Header=BB24_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB24_2
// %bb.10:                              //   in Loop: Header=BB24_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB24_4
.LBB24_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB24_12:
	ret
.Lfunc_end24:
	.size	_ZN9benchmark9quicksortIPddEEvT_S2_, .Lfunc_end24-_ZN9benchmark9quicksortIPddEEvT_S2_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_,"axG",@progbits,_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_,comdat
	.weak	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_ // -- Begin function _ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	.p2align	2
	.type	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_,@function
_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_: // @_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB25_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB25_3
.LBB25_2:                               //   in Loop: Header=BB25_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB25_11
.LBB25_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB25_4 Depth 2
                                        //       Child Loop BB25_5 Depth 3
                                        //       Child Loop BB25_8 Depth 3
	ldr	d0, [x1]
	mov	x4, x1
	mov	x3, x0
.LBB25_4:                               //   Parent Loop BB25_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB25_5 Depth 3
                                        //       Child Loop BB25_8 Depth 3
	sub	x5, x2, x3
.LBB25_5:                               //   Parent Loop BB25_3 Depth=1
                                        //     Parent Loop BB25_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB25_5
// %bb.6:                               //   in Loop: Header=BB25_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB25_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB25_4 Depth=2
	sub	x4, x4, #8
.LBB25_8:                               //   Parent Loop BB25_3 Depth=1
                                        //     Parent Loop BB25_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB25_8
// %bb.9:                               //   in Loop: Header=BB25_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB25_2
// %bb.10:                              //   in Loop: Header=BB25_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB25_4
.LBB25_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB25_12:
	ret
.Lfunc_end25:
	.size	_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_, .Lfunc_end25-_ZN9benchmark9quicksortI14PointerWrapperIdEdEEvT_S3_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_,"axG",@progbits,_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_,comdat
	.weak	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_ // -- Begin function _ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	.p2align	2
	.type	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_,@function
_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_: // @_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB26_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB26_3
.LBB26_2:                               //   in Loop: Header=BB26_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB26_11
.LBB26_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB26_4 Depth 2
                                        //       Child Loop BB26_5 Depth 3
                                        //       Child Loop BB26_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB26_4:                               //   Parent Loop BB26_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB26_5 Depth 3
                                        //       Child Loop BB26_8 Depth 3
	sub	x5, x2, x3
.LBB26_5:                               //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB26_5
// %bb.6:                               //   in Loop: Header=BB26_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB26_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB26_4 Depth=2
	sub	x4, x4, #8
.LBB26_8:                               //   Parent Loop BB26_3 Depth=1
                                        //     Parent Loop BB26_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB26_8
// %bb.9:                               //   in Loop: Header=BB26_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB26_2
// %bb.10:                              //   in Loop: Header=BB26_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB26_4
.LBB26_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB26_12:
	ret
.Lfunc_end26:
	.size	_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_, .Lfunc_end26-_ZN9benchmark9quicksortIP12ValueWrapperIdES2_EEvT_S4_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,"axG",@progbits,_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,comdat
	.weak	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_ // -- Begin function _ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.p2align	2
	.type	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,@function
_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_: // @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB27_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB27_3
.LBB27_2:                               //   in Loop: Header=BB27_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB27_11
.LBB27_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB27_4 Depth 2
                                        //       Child Loop BB27_5 Depth 3
                                        //       Child Loop BB27_8 Depth 3
	ldr	d0, [x1]
	mov	x4, x1
	mov	x3, x0
.LBB27_4:                               //   Parent Loop BB27_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB27_5 Depth 3
                                        //       Child Loop BB27_8 Depth 3
	sub	x5, x2, x3
.LBB27_5:                               //   Parent Loop BB27_3 Depth=1
                                        //     Parent Loop BB27_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB27_5
// %bb.6:                               //   in Loop: Header=BB27_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB27_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB27_4 Depth=2
	sub	x4, x4, #8
.LBB27_8:                               //   Parent Loop BB27_3 Depth=1
                                        //     Parent Loop BB27_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB27_8
// %bb.9:                               //   in Loop: Header=BB27_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB27_2
// %bb.10:                              //   in Loop: Header=BB27_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB27_4
.LBB27_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB27_12:
	ret
.Lfunc_end27:
	.size	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_, .Lfunc_end27-_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,"axG",@progbits,_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,comdat
	.weak	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_ // -- Begin function _ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.p2align	2
	.type	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,@function
_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_: // @_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB28_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB28_3
.LBB28_2:                               //   in Loop: Header=BB28_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB28_11
.LBB28_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB28_4 Depth 2
                                        //       Child Loop BB28_5 Depth 3
                                        //       Child Loop BB28_8 Depth 3
	ldr	d0, [x1]
	mov	x3, x0
	mov	x4, x1
.LBB28_4:                               //   Parent Loop BB28_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB28_5 Depth 3
                                        //       Child Loop BB28_8 Depth 3
	sub	x5, x2, x3
.LBB28_5:                               //   Parent Loop BB28_3 Depth=1
                                        //     Parent Loop BB28_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB28_5
// %bb.6:                               //   in Loop: Header=BB28_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB28_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB28_4 Depth=2
	sub	x4, x4, #8
.LBB28_8:                               //   Parent Loop BB28_3 Depth=1
                                        //     Parent Loop BB28_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB28_8
// %bb.9:                               //   in Loop: Header=BB28_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB28_2
// %bb.10:                              //   in Loop: Header=BB28_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB28_4
.LBB28_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB28_12:
	ret
.Lfunc_end28:
	.size	_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_, .Lfunc_end28-_ZN9benchmark9quicksortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,"axG",@progbits,_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,comdat
	.weak	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_ // -- Begin function _ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	.p2align	2
	.type	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,@function
_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_: // @_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	.cfi_startproc
// %bb.0:
	sub	x2, x1, x1
	cmp	x2, #9
	b.lt	.LBB29_12
// %bb.1:                               // %.preheader1
	stp	x29, x30, [sp, #-16]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 16
	mov	x29, sp
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x0, x1
	sub	x2, x1, #8
	b	.LBB29_3
.LBB29_2:                               //   in Loop: Header=BB29_3 Depth=1
	add	x3, x3, #8
	mov	x0, x1
	mov	x1, x3
	bl	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	cmp	x5, #8
	mov	x1, x3
	b.le	.LBB29_11
.LBB29_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB29_4 Depth 2
                                        //       Child Loop BB29_5 Depth 3
                                        //       Child Loop BB29_8 Depth 3
	ldr	d0, [x1]
	mov	x4, x1
	mov	x3, x0
.LBB29_4:                               //   Parent Loop BB29_3 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB29_5 Depth 3
                                        //       Child Loop BB29_8 Depth 3
	sub	x5, x2, x3
.LBB29_5:                               //   Parent Loop BB29_3 Depth=1
                                        //     Parent Loop BB29_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x3, #-8]!
	add	x5, x5, #8
	fcmp	d0, d1
	b.mi	.LBB29_5
// %bb.6:                               //   in Loop: Header=BB29_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB29_2
// %bb.7:                               // %.preheader
                                        //   in Loop: Header=BB29_4 Depth=2
	sub	x4, x4, #8
.LBB29_8:                               //   Parent Loop BB29_3 Depth=1
                                        //     Parent Loop BB29_4 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d2, [x4, #8]!
	fcmp	d2, d0
	b.mi	.LBB29_8
// %bb.9:                               //   in Loop: Header=BB29_4 Depth=2
	cmp	x4, x3
	b.hs	.LBB29_2
// %bb.10:                              //   in Loop: Header=BB29_4 Depth=2
	str	d2, [x3]
	str	d1, [x4]
	b	.LBB29_4
.LBB29_11:
	.cfi_def_cfa wsp, 16
	ldp	x29, x30, [sp], #16             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
.LBB29_12:
	ret
.Lfunc_end29:
	.size	_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_, .Lfunc_end29-_ZN9benchmark9quicksortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortIPddEEvT_S2_,"axG",@progbits,_ZN9benchmark8heapsortIPddEEvT_S2_,comdat
	.weak	_ZN9benchmark8heapsortIPddEEvT_S2_ // -- Begin function _ZN9benchmark8heapsortIPddEEvT_S2_
	.p2align	2
	.type	_ZN9benchmark8heapsortIPddEEvT_S2_,@function
_ZN9benchmark8heapsortIPddEEvT_S2_:     // @_ZN9benchmark8heapsortIPddEEvT_S2_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB30_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB30_4
.LBB30_2:                               //   in Loop: Header=BB30_4 Depth=1
	mov	x6, x5
.LBB30_3:                               //   in Loop: Header=BB30_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB30_17
.LBB30_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB30_7 Depth 2
                                        //     Child Loop BB30_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB30_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB30_4 Depth=1
	mov	x7, x2
	b	.LBB30_7
.LBB30_6:                               // %select.end
                                        //   in Loop: Header=BB30_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	cmp	x6, x1
	str	d1, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB30_10
.LBB30_7:                               //   Parent Loop BB30_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB30_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB30_7 Depth=2
	add	x6, x6, #1
	b	.LBB30_6
.LBB30_9:                               //   in Loop: Header=BB30_4 Depth=1
	mov	x5, x2
.LBB30_10:                              //   in Loop: Header=BB30_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB30_12
// %bb.11:                              //   in Loop: Header=BB30_4 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	str	d1, [x0, x5, lsl #3]
	mov	x5, x3
.LBB30_12:                              //   in Loop: Header=BB30_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB30_2
.LBB30_13:                              //   Parent Loop BB30_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB30_2
// %bb.14:                              //   in Loop: Header=BB30_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB30_13
	b	.LBB30_3
.LBB30_15:                              //   in Loop: Header=BB30_17 Depth=1
	mov	x3, xzr
.LBB30_16:                              //   in Loop: Header=BB30_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB30_26
.LBB30_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB30_19 Depth 2
                                        //     Child Loop BB30_24 Depth 2
	sub	x2, x1, #1
	ldr	d1, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	d1, [x0, x2, lsl #3]
	b.lo	.LBB30_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB30_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB30_19:                              //   Parent Loop BB30_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	d1, [x0, x3, lsl #3]
	cmp	x4, x2
	str	d1, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB30_19
// %bb.20:                              //   in Loop: Header=BB30_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB30_23
.LBB30_21:                              //   in Loop: Header=BB30_17 Depth=1
	sub	x4, x1, #2
	ldr	d1, [x0, x4, lsl #3]
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB30_24
.LBB30_22:                              //   in Loop: Header=BB30_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB30_21
.LBB30_23:                              //   in Loop: Header=BB30_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB30_16
.LBB30_24:                              //   Parent Loop BB30_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB30_16
// %bb.25:                              //   in Loop: Header=BB30_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB30_24
	b	.LBB30_15
.LBB30_26:
	ret
.Lfunc_end30:
	.size	_ZN9benchmark8heapsortIPddEEvT_S2_, .Lfunc_end30-_ZN9benchmark8heapsortIPddEEvT_S2_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_,"axG",@progbits,_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_,comdat
	.weak	_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_ // -- Begin function _ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_
	.p2align	2
	.type	_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_,@function
_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_: // @_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB31_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB31_4
.LBB31_2:                               //   in Loop: Header=BB31_4 Depth=1
	mov	x6, x5
.LBB31_3:                               //   in Loop: Header=BB31_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB31_17
.LBB31_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB31_7 Depth 2
                                        //     Child Loop BB31_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB31_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB31_4 Depth=1
	mov	x7, x2
	b	.LBB31_7
.LBB31_6:                               // %select.end
                                        //   in Loop: Header=BB31_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	d1, [x0, x5, lsl #3]
	cmp	x6, x1
	str	d1, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB31_10
.LBB31_7:                               //   Parent Loop BB31_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB31_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB31_7 Depth=2
	add	x6, x6, #1
	b	.LBB31_6
.LBB31_9:                               //   in Loop: Header=BB31_4 Depth=1
	mov	x5, x2
.LBB31_10:                              //   in Loop: Header=BB31_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB31_12
// %bb.11:                              //   in Loop: Header=BB31_4 Depth=1
	ldr	d1, [x0, x3, lsl #3]
	str	d1, [x0, x5, lsl #3]
	mov	x5, x3
.LBB31_12:                              //   in Loop: Header=BB31_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB31_2
.LBB31_13:                              //   Parent Loop BB31_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB31_2
// %bb.14:                              //   in Loop: Header=BB31_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB31_13
	b	.LBB31_3
.LBB31_15:                              //   in Loop: Header=BB31_17 Depth=1
	mov	x3, xzr
.LBB31_16:                              //   in Loop: Header=BB31_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB31_26
.LBB31_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB31_19 Depth 2
                                        //     Child Loop BB31_24 Depth 2
	sub	x2, x1, #1
	ldr	d1, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	d1, [x0, x2, lsl #3]
	b.lo	.LBB31_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB31_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB31_19:                              //   Parent Loop BB31_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	d1, [x0, x3, lsl #3]
	cmp	x4, x2
	str	d1, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB31_19
// %bb.20:                              //   in Loop: Header=BB31_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB31_23
.LBB31_21:                              //   in Loop: Header=BB31_17 Depth=1
	sub	x4, x1, #2
	ldr	d1, [x0, x4, lsl #3]
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB31_24
.LBB31_22:                              //   in Loop: Header=BB31_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB31_21
.LBB31_23:                              //   in Loop: Header=BB31_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB31_16
.LBB31_24:                              //   Parent Loop BB31_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB31_16
// %bb.25:                              //   in Loop: Header=BB31_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB31_24
	b	.LBB31_15
.LBB31_26:
	ret
.Lfunc_end31:
	.size	_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_, .Lfunc_end31-_ZN9benchmark8heapsortI14PointerWrapperIdEdEEvT_S3_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_,"axG",@progbits,_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_,comdat
	.weak	_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_ // -- Begin function _ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_
	.p2align	2
	.type	_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_,@function
_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_: // @_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB32_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB32_4
.LBB32_2:                               //   in Loop: Header=BB32_4 Depth=1
	mov	x6, x5
.LBB32_3:                               //   in Loop: Header=BB32_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB32_17
.LBB32_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB32_7 Depth 2
                                        //     Child Loop BB32_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB32_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB32_4 Depth=1
	mov	x7, x2
	b	.LBB32_7
.LBB32_6:                               // %select.end
                                        //   in Loop: Header=BB32_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	x8, [x0, x5, lsl #3]
	cmp	x6, x1
	str	x8, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB32_10
.LBB32_7:                               //   Parent Loop BB32_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB32_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB32_7 Depth=2
	add	x6, x6, #1
	b	.LBB32_6
.LBB32_9:                               //   in Loop: Header=BB32_4 Depth=1
	mov	x5, x2
.LBB32_10:                              //   in Loop: Header=BB32_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB32_12
// %bb.11:                              //   in Loop: Header=BB32_4 Depth=1
	ldr	x6, [x0, x3, lsl #3]
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
.LBB32_12:                              //   in Loop: Header=BB32_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB32_2
.LBB32_13:                              //   Parent Loop BB32_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB32_2
// %bb.14:                              //   in Loop: Header=BB32_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB32_13
	b	.LBB32_3
.LBB32_15:                              //   in Loop: Header=BB32_17 Depth=1
	mov	x3, xzr
.LBB32_16:                              //   in Loop: Header=BB32_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB32_26
.LBB32_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB32_19 Depth 2
                                        //     Child Loop BB32_24 Depth 2
	sub	x2, x1, #1
	ldr	x3, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	x3, [x0, x2, lsl #3]
	b.lo	.LBB32_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB32_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB32_19:                              //   Parent Loop BB32_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	x6, [x0, x3, lsl #3]
	cmp	x4, x2
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB32_19
// %bb.20:                              //   in Loop: Header=BB32_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB32_23
.LBB32_21:                              //   in Loop: Header=BB32_17 Depth=1
	sub	x4, x1, #2
	ldr	x5, [x0, x4, lsl #3]
	str	x5, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB32_24
.LBB32_22:                              //   in Loop: Header=BB32_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB32_21
.LBB32_23:                              //   in Loop: Header=BB32_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB32_16
.LBB32_24:                              //   Parent Loop BB32_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB32_16
// %bb.25:                              //   in Loop: Header=BB32_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB32_24
	b	.LBB32_15
.LBB32_26:
	ret
.Lfunc_end32:
	.size	_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_, .Lfunc_end32-_ZN9benchmark8heapsortIP12ValueWrapperIdES2_EEvT_S4_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,"axG",@progbits,_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,comdat
	.weak	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_ // -- Begin function _ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.p2align	2
	.type	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_,@function
_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_: // @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB33_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB33_4
.LBB33_2:                               //   in Loop: Header=BB33_4 Depth=1
	mov	x6, x5
.LBB33_3:                               //   in Loop: Header=BB33_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB33_17
.LBB33_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB33_7 Depth 2
                                        //     Child Loop BB33_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB33_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB33_4 Depth=1
	mov	x7, x2
	b	.LBB33_7
.LBB33_6:                               // %select.end
                                        //   in Loop: Header=BB33_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	x8, [x0, x5, lsl #3]
	cmp	x6, x1
	str	x8, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB33_10
.LBB33_7:                               //   Parent Loop BB33_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB33_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB33_7 Depth=2
	add	x6, x6, #1
	b	.LBB33_6
.LBB33_9:                               //   in Loop: Header=BB33_4 Depth=1
	mov	x5, x2
.LBB33_10:                              //   in Loop: Header=BB33_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB33_12
// %bb.11:                              //   in Loop: Header=BB33_4 Depth=1
	ldr	x6, [x0, x3, lsl #3]
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
.LBB33_12:                              //   in Loop: Header=BB33_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB33_2
.LBB33_13:                              //   Parent Loop BB33_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB33_2
// %bb.14:                              //   in Loop: Header=BB33_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB33_13
	b	.LBB33_3
.LBB33_15:                              //   in Loop: Header=BB33_17 Depth=1
	mov	x3, xzr
.LBB33_16:                              //   in Loop: Header=BB33_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB33_26
.LBB33_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB33_19 Depth 2
                                        //     Child Loop BB33_24 Depth 2
	sub	x2, x1, #1
	ldr	x3, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	x3, [x0, x2, lsl #3]
	b.lo	.LBB33_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB33_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB33_19:                              //   Parent Loop BB33_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	x6, [x0, x3, lsl #3]
	cmp	x4, x2
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB33_19
// %bb.20:                              //   in Loop: Header=BB33_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB33_23
.LBB33_21:                              //   in Loop: Header=BB33_17 Depth=1
	sub	x4, x1, #2
	ldr	x5, [x0, x4, lsl #3]
	str	x5, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB33_24
.LBB33_22:                              //   in Loop: Header=BB33_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB33_21
.LBB33_23:                              //   in Loop: Header=BB33_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB33_16
.LBB33_24:                              //   Parent Loop BB33_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB33_16
// %bb.25:                              //   in Loop: Header=BB33_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB33_24
	b	.LBB33_15
.LBB33_26:
	ret
.Lfunc_end33:
	.size	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_, .Lfunc_end33-_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIdEES3_EEvT_S5_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,"axG",@progbits,_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,comdat
	.weak	_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_ // -- Begin function _ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.p2align	2
	.type	_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_,@function
_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_: // @_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB34_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB34_4
.LBB34_2:                               //   in Loop: Header=BB34_4 Depth=1
	mov	x6, x5
.LBB34_3:                               //   in Loop: Header=BB34_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB34_17
.LBB34_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB34_7 Depth 2
                                        //     Child Loop BB34_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB34_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB34_4 Depth=1
	mov	x7, x2
	b	.LBB34_7
.LBB34_6:                               // %select.end
                                        //   in Loop: Header=BB34_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	x8, [x0, x5, lsl #3]
	cmp	x6, x1
	str	x8, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB34_10
.LBB34_7:                               //   Parent Loop BB34_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB34_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB34_7 Depth=2
	add	x6, x6, #1
	b	.LBB34_6
.LBB34_9:                               //   in Loop: Header=BB34_4 Depth=1
	mov	x5, x2
.LBB34_10:                              //   in Loop: Header=BB34_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB34_12
// %bb.11:                              //   in Loop: Header=BB34_4 Depth=1
	ldr	x6, [x0, x3, lsl #3]
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
.LBB34_12:                              //   in Loop: Header=BB34_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB34_2
.LBB34_13:                              //   Parent Loop BB34_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB34_2
// %bb.14:                              //   in Loop: Header=BB34_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB34_13
	b	.LBB34_3
.LBB34_15:                              //   in Loop: Header=BB34_17 Depth=1
	mov	x3, xzr
.LBB34_16:                              //   in Loop: Header=BB34_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB34_26
.LBB34_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB34_19 Depth 2
                                        //     Child Loop BB34_24 Depth 2
	sub	x2, x1, #1
	ldr	x3, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	x3, [x0, x2, lsl #3]
	b.lo	.LBB34_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB34_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB34_19:                              //   Parent Loop BB34_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	x6, [x0, x3, lsl #3]
	cmp	x4, x2
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB34_19
// %bb.20:                              //   in Loop: Header=BB34_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB34_23
.LBB34_21:                              //   in Loop: Header=BB34_17 Depth=1
	sub	x4, x1, #2
	ldr	x5, [x0, x4, lsl #3]
	str	x5, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB34_24
.LBB34_22:                              //   in Loop: Header=BB34_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB34_21
.LBB34_23:                              //   in Loop: Header=BB34_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB34_16
.LBB34_24:                              //   Parent Loop BB34_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB34_16
// %bb.25:                              //   in Loop: Header=BB34_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB34_24
	b	.LBB34_15
.LBB34_26:
	ret
.Lfunc_end34:
	.size	_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_, .Lfunc_end34-_ZN9benchmark8heapsortIP12ValueWrapperIS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IS1_IdEEEEEEEEEESB_EEvT_SD_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,"axG",@progbits,_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,comdat
	.weak	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_ // -- Begin function _ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	.p2align	2
	.type	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_,@function
_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_: // @_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
	.cfi_startproc
// %bb.0:
	sub	x1, x1, x0
	asr	x1, x1, #3
	cmp	x1, #2
	b.lt	.LBB35_26
// %bb.1:
	lsr	x2, x1, #1
	sub	x3, x1, #1
	b	.LBB35_4
.LBB35_2:                               //   in Loop: Header=BB35_4 Depth=1
	mov	x6, x5
.LBB35_3:                               //   in Loop: Header=BB35_4 Depth=1
	cmp	x4, #1
	str	d0, [x0, x6, lsl #3]
	b.le	.LBB35_17
.LBB35_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB35_7 Depth 2
                                        //     Child Loop BB35_13 Depth 2
	mov	x4, x2
	sub	x2, x2, #1
	lsl	x5, x2, #1
	ldr	d0, [x0, x2, lsl #3]
	add	x6, x5, #2
	cmp	x6, x1
	b.ge	.LBB35_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB35_4 Depth=1
	mov	x7, x2
	b	.LBB35_7
.LBB35_6:                               // %select.end
                                        //   in Loop: Header=BB35_7 Depth=2
	sub	x5, x6, #1
	lsl	x6, x6, #1
	ldr	x8, [x0, x5, lsl #3]
	cmp	x6, x1
	str	x8, [x0, x7, lsl #3]
	mov	x7, x5
	b.ge	.LBB35_10
.LBB35_7:                               //   Parent Loop BB35_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x5, x0, x6, lsl #3
	ldp	d1, d2, [x5, #-8]
	fcmp	d1, d2
	b.pl	.LBB35_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB35_7 Depth=2
	add	x6, x6, #1
	b	.LBB35_6
.LBB35_9:                               //   in Loop: Header=BB35_4 Depth=1
	mov	x5, x2
.LBB35_10:                              //   in Loop: Header=BB35_4 Depth=1
	cmp	x6, x1
	b.ne	.LBB35_12
// %bb.11:                              //   in Loop: Header=BB35_4 Depth=1
	ldr	x6, [x0, x3, lsl #3]
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
.LBB35_12:                              //   in Loop: Header=BB35_4 Depth=1
	cmp	x5, x4
	b.lt	.LBB35_2
.LBB35_13:                              //   Parent Loop BB35_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x6, x5, #1
	add	x6, x6, x6, lsr #63
	asr	x6, x6, #1
	ldr	d1, [x0, x6, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB35_2
// %bb.14:                              //   in Loop: Header=BB35_13 Depth=2
	cmp	x6, x4
	str	d1, [x0, x5, lsl #3]
	mov	x5, x6
	b.ge	.LBB35_13
	b	.LBB35_3
.LBB35_15:                              //   in Loop: Header=BB35_17 Depth=1
	mov	x3, xzr
.LBB35_16:                              //   in Loop: Header=BB35_17 Depth=1
	cmp	x1, #2
	mov	x1, x2
	str	d0, [x0, x3, lsl #3]
	b.le	.LBB35_26
.LBB35_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB35_19 Depth 2
                                        //     Child Loop BB35_24 Depth 2
	sub	x2, x1, #1
	ldr	x3, [x0]
	ldr	d0, [x0, x2, lsl #3]
	cmp	x2, #3
	str	x3, [x0, x2, lsl #3]
	b.lo	.LBB35_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB35_17 Depth=1
	mov	x5, xzr
	mov	w4, #2                          // =0x2
.LBB35_19:                              //   Parent Loop BB35_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x3, x0, x4, lsl #3
	ldp	d1, d2, [x3, #-8]
	fcmp	d1, d2
	cinc	x3, x4, mi
	lsl	x4, x3, #1
	sub	x3, x3, #1
	ldr	x6, [x0, x3, lsl #3]
	cmp	x4, x2
	str	x6, [x0, x5, lsl #3]
	mov	x5, x3
	b.lt	.LBB35_19
// %bb.20:                              //   in Loop: Header=BB35_17 Depth=1
	cmp	x4, x2
	b.ne	.LBB35_23
.LBB35_21:                              //   in Loop: Header=BB35_17 Depth=1
	sub	x4, x1, #2
	ldr	x5, [x0, x4, lsl #3]
	str	x5, [x0, x3, lsl #3]
	mov	x3, x4
	b	.LBB35_24
.LBB35_22:                              //   in Loop: Header=BB35_17 Depth=1
	mov	x3, xzr
	mov	w4, #2                          // =0x2
	cmp	x4, x2
	b.eq	.LBB35_21
.LBB35_23:                              //   in Loop: Header=BB35_17 Depth=1
	cmp	x3, #1
	b.lt	.LBB35_16
.LBB35_24:                              //   Parent Loop BB35_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x5, x3, #1
	lsr	x4, x5, #1
	ldr	d1, [x0, x4, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB35_16
// %bb.25:                              //   in Loop: Header=BB35_24 Depth=2
	cmp	x5, #1
	str	d1, [x0, x3, lsl #3]
	mov	x3, x4
	b.hi	.LBB35_24
	b	.LBB35_15
.LBB35_26:
	ret
.Lfunc_end35:
	.size	_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_, .Lfunc_end35-_ZN9benchmark8heapsortI14PointerWrapperI12ValueWrapperIS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IS2_IdEEEEEEEEEEESC_EEvT_SE_
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

	.type	iterations,@object              // @iterations
	.data
	.globl	iterations
	.p2align	2, 0x0
iterations:
	.word	200000                          // 0x30d40
	.size	iterations, 4

	.type	init_value,@object              // @init_value
	.globl	init_value
	.p2align	3, 0x0
init_value:
	.xword	0x4008000000000000              // double 3
	.size	init_value, 8

	.type	data,@object                    // @data
	.bss
	.globl	data
	.p2align	3, 0x0
data:
	.zero	16000
	.size	data, 16000

	.type	VData,@object                   // @VData
	.globl	VData
	.p2align	3, 0x0
VData:
	.zero	16000
	.size	VData, 16000

	.type	V10Data,@object                 // @V10Data
	.globl	V10Data
	.p2align	3, 0x0
V10Data:
	.zero	16000
	.size	V10Data, 16000

	.type	dataMaster,@object              // @dataMaster
	.globl	dataMaster
	.p2align	3, 0x0
dataMaster:
	.zero	16000
	.size	dataMaster, 16000

	.type	VDataMaster,@object             // @VDataMaster
	.globl	VDataMaster
	.p2align	3, 0x0
VDataMaster:
	.zero	16000
	.size	VDataMaster, 16000

	.type	V10DataMaster,@object           // @V10DataMaster
	.globl	V10DataMaster
	.p2align	3, 0x0
V10DataMaster:
	.zero	16000
	.size	V10DataMaster, 16000

	.type	dpb,@object                     // @dpb
	.data
	.globl	dpb
	.p2align	3, 0x0
dpb:
	.xword	data
	.size	dpb, 8

	.type	dpe,@object                     // @dpe
	.globl	dpe
	.p2align	3, 0x0
dpe:
	.xword	data+16000
	.size	dpe, 8

	.type	dMpb,@object                    // @dMpb
	.globl	dMpb
	.p2align	3, 0x0
dMpb:
	.xword	dataMaster
	.size	dMpb, 8

	.type	dMpe,@object                    // @dMpe
	.globl	dMpe
	.p2align	3, 0x0
dMpe:
	.xword	dataMaster+16000
	.size	dMpe, 8

	.type	DVpb,@object                    // @DVpb
	.globl	DVpb
	.p2align	3, 0x0
DVpb:
	.xword	VData
	.size	DVpb, 8

	.type	DVpe,@object                    // @DVpe
	.globl	DVpe
	.p2align	3, 0x0
DVpe:
	.xword	VData+16000
	.size	DVpe, 8

	.type	DVMpb,@object                   // @DVMpb
	.globl	DVMpb
	.p2align	3, 0x0
DVMpb:
	.xword	VDataMaster
	.size	DVMpb, 8

	.type	DVMpe,@object                   // @DVMpe
	.globl	DVMpe
	.p2align	3, 0x0
DVMpe:
	.xword	VDataMaster+16000
	.size	DVMpe, 8

	.type	DV10pb,@object                  // @DV10pb
	.globl	DV10pb
	.p2align	3, 0x0
DV10pb:
	.xword	V10Data
	.size	DV10pb, 8

	.type	DV10pe,@object                  // @DV10pe
	.globl	DV10pe
	.p2align	3, 0x0
DV10pe:
	.xword	V10Data+16000
	.size	DV10pe, 8

	.type	DV10Mpb,@object                 // @DV10Mpb
	.globl	DV10Mpb
	.p2align	3, 0x0
DV10Mpb:
	.xword	V10DataMaster
	.size	DV10Mpb, 8

	.type	DV10Mpe,@object                 // @DV10Mpe
	.globl	DV10Mpe
	.p2align	3, 0x0
DV10Mpe:
	.xword	V10DataMaster+16000
	.size	DV10Mpe, 8

	.type	dPb,@object                     // @dPb
	.globl	dPb
	.p2align	3, 0x0
dPb:
	.xword	data
	.size	dPb, 8

	.type	dPe,@object                     // @dPe
	.globl	dPe
	.p2align	3, 0x0
dPe:
	.xword	data+16000
	.size	dPe, 8

	.type	dMPb,@object                    // @dMPb
	.globl	dMPb
	.p2align	3, 0x0
dMPb:
	.xword	dataMaster
	.size	dMPb, 8

	.type	dMPe,@object                    // @dMPe
	.globl	dMPe
	.p2align	3, 0x0
dMPe:
	.xword	dataMaster+16000
	.size	dMPe, 8

	.type	DVPb,@object                    // @DVPb
	.globl	DVPb
	.p2align	3, 0x0
DVPb:
	.xword	VData
	.size	DVPb, 8

	.type	DVPe,@object                    // @DVPe
	.globl	DVPe
	.p2align	3, 0x0
DVPe:
	.xword	VData+16000
	.size	DVPe, 8

	.type	DVMPb,@object                   // @DVMPb
	.globl	DVMPb
	.p2align	3, 0x0
DVMPb:
	.xword	VDataMaster
	.size	DVMPb, 8

	.type	DVMPe,@object                   // @DVMPe
	.globl	DVMPe
	.p2align	3, 0x0
DVMPe:
	.xword	VDataMaster+16000
	.size	DVMPe, 8

	.type	DV10Pb,@object                  // @DV10Pb
	.globl	DV10Pb
	.p2align	3, 0x0
DV10Pb:
	.xword	V10Data
	.size	DV10Pb, 8

	.type	DV10Pe,@object                  // @DV10Pe
	.globl	DV10Pe
	.p2align	3, 0x0
DV10Pe:
	.xword	V10Data+16000
	.size	DV10Pe, 8

	.type	DV10MPb,@object                 // @DV10MPb
	.globl	DV10MPb
	.p2align	3, 0x0
DV10MPb:
	.xword	V10DataMaster
	.size	DV10MPb, 8

	.type	DV10MPe,@object                 // @DV10MPe
	.globl	DV10MPe
	.p2align	3, 0x0
DV10MPe:
	.xword	V10DataMaster+16000
	.size	DV10MPe, 8

	.type	.L.str.32,@object               // @.str.32
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.32:
	.asciz	"insertion_sort double pointer"
	.size	.L.str.32, 30

	.type	.L.str.33,@object               // @.str.33
.L.str.33:
	.asciz	"insertion_sort double pointer_class"
	.size	.L.str.33, 36

	.type	.L.str.34,@object               // @.str.34
.L.str.34:
	.asciz	"insertion_sort DoubleValueWrapper pointer"
	.size	.L.str.34, 42

	.type	.L.str.35,@object               // @.str.35
.L.str.35:
	.asciz	"insertion_sort DoubleValueWrapper pointer_class"
	.size	.L.str.35, 48

	.type	.L.str.36,@object               // @.str.36
.L.str.36:
	.asciz	"insertion_sort DoubleValueWrapper10 pointer"
	.size	.L.str.36, 44

	.type	.L.str.37,@object               // @.str.37
.L.str.37:
	.asciz	"insertion_sort DoubleValueWrapper10 pointer_class"
	.size	.L.str.37, 50

	.type	.L.str.38,@object               // @.str.38
.L.str.38:
	.asciz	"quicksort double pointer"
	.size	.L.str.38, 25

	.type	.L.str.39,@object               // @.str.39
.L.str.39:
	.asciz	"quicksort double pointer_class"
	.size	.L.str.39, 31

	.type	.L.str.40,@object               // @.str.40
.L.str.40:
	.asciz	"quicksort DoubleValueWrapper pointer"
	.size	.L.str.40, 37

	.type	.L.str.41,@object               // @.str.41
.L.str.41:
	.asciz	"quicksort DoubleValueWrapper pointer_class"
	.size	.L.str.41, 43

	.type	.L.str.42,@object               // @.str.42
.L.str.42:
	.asciz	"quicksort DoubleValueWrapper10 pointer"
	.size	.L.str.42, 39

	.type	.L.str.43,@object               // @.str.43
.L.str.43:
	.asciz	"quicksort DoubleValueWrapper10 pointer_class"
	.size	.L.str.43, 45

	.type	.L.str.44,@object               // @.str.44
.L.str.44:
	.asciz	"heap_sort double pointer"
	.size	.L.str.44, 25

	.type	.L.str.45,@object               // @.str.45
.L.str.45:
	.asciz	"heap_sort double pointer_class"
	.size	.L.str.45, 31

	.type	.L.str.46,@object               // @.str.46
.L.str.46:
	.asciz	"heap_sort DoubleValueWrapper pointer"
	.size	.L.str.46, 37

	.type	.L.str.47,@object               // @.str.47
.L.str.47:
	.asciz	"heap_sort DoubleValueWrapper pointer_class"
	.size	.L.str.47, 43

	.type	.L.str.48,@object               // @.str.48
.L.str.48:
	.asciz	"heap_sort DoubleValueWrapper10 pointer"
	.size	.L.str.48, 39

	.type	.L.str.49,@object               // @.str.49
.L.str.49:
	.asciz	"heap_sort DoubleValueWrapper10 pointer_class"
	.size	.L.str.49, 45

	.type	.L.str.50,@object               // @.str.50
.L.str.50:
	.asciz	"test %i failed\n"
	.size	.L.str.50, 16

	.type	.L.str.51,@object               // @.str.51
.L.str.51:
	.asciz	"sort test %i failed\n"
	.size	.L.str.51, 21

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
