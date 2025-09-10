	.file	"stepanov_vector.cpp"
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
	.globl	main                            // -- Begin function main
	.p2align	2
	.type	main,@function
main:                                   // @main
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 156, DW.ref.__gxx_personality_v0
	.cfi_lsda 28, .Lexception0
// %bb.0:
	sub	sp, sp, #208
	.cfi_def_cfa_offset 208
	str	d8, [sp, #96]                   // 8-byte Folded Spill
	stp	x29, x30, [sp, #112]            // 16-byte Folded Spill
	stp	x28, x27, [sp, #128]            // 16-byte Folded Spill
	stp	x26, x25, [sp, #144]            // 16-byte Folded Spill
	stp	x24, x23, [sp, #160]            // 16-byte Folded Spill
	stp	x22, x21, [sp, #176]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #192]            // 16-byte Folded Spill
	add	x29, sp, #112
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
	.cfi_remember_state
	cmp	w0, #2
	adrp	x25, iterations
	adrp	x21, init_value
	b.lt	.LBB5_3
// %bb.1:
	mov	w20, w0
	ldr	x0, [x1, #8]
	mov	x19, x1
	mov	x1, xzr
	mov	w2, #10                         // =0xa
	bl	__isoc23_strtol
	cmp	w20, #2
	str	w0, [x25, :lo12:iterations]
	b.eq	.LBB5_3
// %bb.2:
	ldr	x0, [x19, #16]
	mov	x1, xzr
	bl	strtod
	str	d0, [x21, :lo12:init_value]
.LBB5_3:
	ldr	d0, [x21, :lo12:init_value]
	fcvtzs	w8, d0
	add	w0, w8, #123
	bl	srand
	adrp	x27, dpb
	adrp	x8, dpe
	ldr	d1, [x21, :lo12:init_value]
	ldr	x22, [x27, :lo12:dpb]
	ldr	x23, [x8, :lo12:dpe]
	cmp	x22, x23
	b.eq	.LBB5_10
// %bb.4:
	sub	x8, x23, x22
	sub	x9, x8, #8
	mov	x8, x22
	cmp	x9, #24
	b.lo	.LBB5_8
// %bb.5:
	lsr	x8, x9, #3
	dup	v0.2d, v1.d[0]
	add	x11, x22, #16
	add	x9, x8, #1
	and	x10, x9, #0x3ffffffffffffffc
	add	x8, x22, x10, lsl #3
	mov	x12, x10
.LBB5_6:                                // =>This Inner Loop Header: Depth=1
	subs	x12, x12, #4
	stp	q0, q0, [x11, #-16]
	add	x11, x11, #32
	b.ne	.LBB5_6
// %bb.7:
	cmp	x9, x10
	b.eq	.LBB5_9
.LBB5_8:                                // =>This Inner Loop Header: Depth=1
	str	d1, [x8], #8
	cmp	x8, x23
	b.ne	.LBB5_8
.LBB5_9:
	ldr	d0, [x21, :lo12:init_value]
	str	q0, [sp, #48]                   // 16-byte Folded Spill
	b	.LBB5_11
.LBB5_10:
	str	q1, [sp, #48]                   // 16-byte Folded Spill
.LBB5_11:
	mov	w0, #16000                      // =0x3e80
	bl	_Znwm
	mov	w1, wzr
	mov	w2, #16000                      // =0x3e80
	mov	x19, x0
	bl	memset
	ldr	q1, [sp, #48]                   // 16-byte Folded Reload
	mov	x8, #-16000                     // =0xffffffffffffc180
	dup	v0.2d, v1.d[0]
.LBB5_12:                               // =>This Inner Loop Header: Depth=1
	add	x9, x19, x8
	adds	x8, x8, #32
	str	q0, [x9, #16000]
	str	q0, [x9, #16016]
	b.ne	.LBB5_12
// %bb.13:
	ldr	w8, [x25, :lo12:iterations]
	adrp	x26, current_test
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.14:
	cmp	x22, x23
	b.eq	.LBB5_21
// %bb.15:                              // %.preheader90
	mov	x9, #70368744177664             // =0x400000000000
	mov	w24, wzr
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	movk	x9, #16543, lsl #48
	fmov	d8, x9
	b	.LBB5_17
.LBB5_16:                               //   in Loop: Header=BB5_17 Depth=1
	add	w24, w24, #1
	cmp	w24, w8
	b.ge	.LBB5_25
.LBB5_17:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_18 Depth 2
	movi	d0, #0000000000000000
	mov	x9, x22
.LBB5_18:                               //   Parent Loop BB5_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x9], #8
	cmp	x9, x23
	fadd	d0, d0, d1
	b.ne	.LBB5_18
// %bb.19:                              //   in Loop: Header=BB5_17 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_16
// %bb.20:                              //   in Loop: Header=BB5_17 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_16
.LBB5_21:                               // %.preheader88
	mov	x9, #70368744177664             // =0x400000000000
	mov	w22, wzr
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	movk	x9, #16543, lsl #48
	fmov	d8, x9
	b	.LBB5_23
.LBB5_22:                               //   in Loop: Header=BB5_23 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_25
.LBB5_23:                               // =>This Inner Loop Header: Depth=1
	fmul	d0, d1, d8
	fcmp	d0, #0.0
	b.eq	.LBB5_22
// %bb.24:                              //   in Loop: Header=BB5_23 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	d1, [x21, :lo12:init_value]
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_22
.LBB5_25:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.26:                              // %.preheader87
	mov	x9, #70368744177664             // =0x400000000000
	mov	w22, wzr
	add	x23, x19, #16
	movk	x9, #16543, lsl #48
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x9
	b	.LBB5_28
.LBB5_27:                               //   in Loop: Header=BB5_28 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_32
.LBB5_28:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_29 Depth 2
	movi	d0, #0000000000000000
	mov	x9, x23
	mov	w10, #2000                      // =0x7d0
.LBB5_29:                               //   Parent Loop BB5_28 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q2, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	mov	d3, v1.d[1]
	fadd	d0, d0, d1
	mov	d1, v2.d[1]
	fadd	d0, d0, d3
	fadd	d0, d0, d2
	fadd	d0, d0, d1
	b.ne	.LBB5_29
// %bb.30:                              //   in Loop: Header=BB5_28 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_27
// %bb.31:                              //   in Loop: Header=BB5_28 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_27
.LBB5_32:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.33:
	adrp	x9, rdpb
	mov	w22, wzr
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	ldr	x23, [x9, :lo12:rdpb]
	adrp	x9, rdpe
	ldr	x24, [x9, :lo12:rdpe]
	mov	x9, #70368744177664             // =0x400000000000
	movk	x9, #16543, lsl #48
	fmov	d8, x9
	b	.LBB5_35
.LBB5_34:                               //   in Loop: Header=BB5_35 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_40
.LBB5_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_37 Depth 2
	movi	d0, #0000000000000000
	cmp	x23, x24
	b.eq	.LBB5_38
// %bb.36:                              // %.preheader85
                                        //   in Loop: Header=BB5_35 Depth=1
	mov	x9, x23
.LBB5_37:                               //   Parent Loop BB5_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x9, #-8]!
	cmp	x9, x24
	fadd	d0, d0, d1
	b.ne	.LBB5_37
.LBB5_38:                               //   in Loop: Header=BB5_35 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_34
// %bb.39:                              //   in Loop: Header=BB5_35 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_34
.LBB5_40:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.41:                              // %.preheader84
	mov	x10, #70368744177664            // =0x400000000000
	mov	w9, #15992                      // =0x3e78
	mov	w22, wzr
	movk	x10, #16543, lsl #48
	add	x23, x19, x9
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x10
	b	.LBB5_43
.LBB5_42:                               //   in Loop: Header=BB5_43 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_47
.LBB5_43:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_44 Depth 2
	movi	d0, #0000000000000000
	mov	x9, #-16000                     // =0xffffffffffffc180
	mov	x10, x23
.LBB5_44:                               //   Parent Loop BB5_43 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x10], #-8
	adds	x9, x9, #8
	fadd	d0, d0, d1
	b.ne	.LBB5_44
// %bb.45:                              //   in Loop: Header=BB5_43 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_42
// %bb.46:                              //   in Loop: Header=BB5_43 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_42
.LBB5_47:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.48:                              // %.preheader83
	mov	x10, #70368744177664            // =0x400000000000
	mov	w9, #15992                      // =0x3e78
	mov	w22, wzr
	movk	x10, #16543, lsl #48
	add	x23, x19, x9
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x10
	b	.LBB5_50
.LBB5_49:                               //   in Loop: Header=BB5_50 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_54
.LBB5_50:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_51 Depth 2
	movi	d0, #0000000000000000
	mov	x9, #-16000                     // =0xffffffffffffc180
	mov	x10, x23
.LBB5_51:                               //   Parent Loop BB5_50 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x10], #-8
	adds	x9, x9, #8
	fadd	d0, d0, d1
	b.ne	.LBB5_51
// %bb.52:                              //   in Loop: Header=BB5_50 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_49
// %bb.53:                              //   in Loop: Header=BB5_50 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_49
.LBB5_54:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.55:
	adrp	x9, rrdpb+8
	adrp	x10, rrdpe+8
	mov	w22, wzr
	ldr	x23, [x9, :lo12:rrdpb+8]
	mov	x9, #70368744177664             // =0x400000000000
	ldr	x24, [x10, :lo12:rrdpe+8]
	movk	x9, #16543, lsl #48
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x9
	b	.LBB5_57
.LBB5_56:                               //   in Loop: Header=BB5_57 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_62
.LBB5_57:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_59 Depth 2
	movi	d0, #0000000000000000
	cmp	x23, x24
	b.eq	.LBB5_60
// %bb.58:                              // %.preheader81
                                        //   in Loop: Header=BB5_57 Depth=1
	mov	x9, x23
.LBB5_59:                               //   Parent Loop BB5_57 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x9], #8
	cmp	x9, x24
	fadd	d0, d0, d1
	b.ne	.LBB5_59
.LBB5_60:                               //   in Loop: Header=BB5_57 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_56
// %bb.61:                              //   in Loop: Header=BB5_57 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_56
.LBB5_62:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.63:                              // %.preheader80
	mov	x9, #70368744177664             // =0x400000000000
	mov	w22, wzr
	add	x23, x19, #16
	movk	x9, #16543, lsl #48
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x9
	b	.LBB5_65
.LBB5_64:                               //   in Loop: Header=BB5_65 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_69
.LBB5_65:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_66 Depth 2
	movi	d0, #0000000000000000
	mov	x9, x23
	mov	w10, #2000                      // =0x7d0
.LBB5_66:                               //   Parent Loop BB5_65 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q2, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	mov	d3, v1.d[1]
	fadd	d0, d0, d1
	mov	d1, v2.d[1]
	fadd	d0, d0, d3
	fadd	d0, d0, d2
	fadd	d0, d0, d1
	b.ne	.LBB5_66
// %bb.67:                              //   in Loop: Header=BB5_65 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_64
// %bb.68:                              //   in Loop: Header=BB5_65 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_64
.LBB5_69:
	cmp	w8, #1
	b.lt	.LBB5_76
// %bb.70:                              // %.preheader78
	mov	x9, #70368744177664             // =0x400000000000
	mov	w22, wzr
	add	x23, x19, #16
	movk	x9, #16543, lsl #48
	adrp	x20, .L.str.51
	add	x20, x20, :lo12:.L.str.51
	fmov	d8, x9
	b	.LBB5_72
.LBB5_71:                               //   in Loop: Header=BB5_72 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB5_76
.LBB5_72:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_73 Depth 2
	movi	d0, #0000000000000000
	mov	x9, x23
	mov	w10, #2000                      // =0x7d0
.LBB5_73:                               //   Parent Loop BB5_72 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q2, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	mov	d3, v1.d[1]
	fadd	d0, d0, d1
	mov	d1, v2.d[1]
	fadd	d0, d0, d3
	fadd	d0, d0, d2
	fadd	d0, d0, d1
	b.ne	.LBB5_73
// %bb.74:                              //   in Loop: Header=BB5_72 Depth=1
	ldr	d1, [x21, :lo12:init_value]
	fmul	d1, d1, d8
	fcmp	d0, d1
	b.eq	.LBB5_71
// %bb.75:                              //   in Loop: Header=BB5_72 Depth=1
	ldr	w1, [x26, :lo12:current_test]
	mov	x0, x20
	bl	printf
	ldr	w8, [x25, :lo12:iterations]
	b	.LBB5_71
.LBB5_76:
	mov	w9, #19923                      // =0x4dd3
	movk	w9, #4194, lsl #16
	smull	x8, w8, w9
	asr	x8, x8, #38
	add	w8, w8, w8, lsr #31
	str	w8, [x25, :lo12:iterations]
.Ltmp0:                                 // EH_LABEL
	mov	w0, #16000                      // =0x3e80
	bl	_Znwm
.Ltmp1:                                 // EH_LABEL
// %bb.77:
	mov	w1, wzr
	mov	w2, #16000                      // =0x3e80
	mov	x28, x0
	mov	w20, #16000                     // =0x3e80
	bl	memset
	adrp	x8, dMpb
	adrp	x9, dMpe
	ldr	x21, [x8, :lo12:dMpb]
	ldr	x22, [x9, :lo12:dMpe]
	cmp	x21, x22
	b.eq	.LBB5_80
.LBB5_78:                               // =>This Inner Loop Header: Depth=1
	bl	rand
	scvtf	d0, w0
	str	d0, [x21], #8
	cmp	x21, x22
	b.ne	.LBB5_78
// %bb.79:
	adrp	x8, dMpb
	ldr	x21, [x8, :lo12:dMpb]
	adrp	x8, dMpe
	ldr	x22, [x8, :lo12:dMpe]
.LBB5_80:
	cmp	x21, x22
	b.eq	.LBB5_87
// %bb.81:
	sub	x8, x22, x21
	mov	x9, x28
	sub	x10, x8, #8
	mov	x8, x21
	cmp	x10, #24
	b.lo	.LBB5_86
// %bb.82:
	sub	x8, x28, x21
	mov	x9, x28
	cmp	x8, #32
	mov	x8, x21
	b.lo	.LBB5_86
// %bb.83:
	lsr	x8, x10, #3
	add	x12, x28, #16
	add	x13, x21, #16
	add	x10, x8, #1
	and	x11, x10, #0x3ffffffffffffffc
	lsl	x9, x11, #3
	mov	x14, x11
	add	x8, x21, x9
	add	x9, x28, x9
.LBB5_84:                               // =>This Inner Loop Header: Depth=1
	ldp	q0, q1, [x13, #-16]
	subs	x14, x14, #4
	add	x13, x13, #32
	stp	q0, q1, [x12, #-16]
	add	x12, x12, #32
	b.ne	.LBB5_84
// %bb.85:
	cmp	x10, x11
	b.eq	.LBB5_87
.LBB5_86:                               // =>This Inner Loop Header: Depth=1
	ldr	d0, [x8], #8
	cmp	x8, x22
	str	d0, [x9], #8
	b.ne	.LBB5_86
.LBB5_87:
	adrp	x8, dpe
	ldr	x2, [x27, :lo12:dpb]
	ldr	x3, [x8, :lo12:dpe]
.Ltmp3:                                 // EH_LABEL
	movi	d0, #0000000000000000
	adrp	x4, .L.str.26
	add	x4, x4, :lo12:.L.str.26
	mov	x0, x21
	mov	x1, x22
	bl	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
.Ltmp4:                                 // EH_LABEL
// %bb.88:
.Ltmp5:                                 // EH_LABEL
	movi	d0, #0000000000000000
	mov	w8, #16000                      // =0x3e80
	adrp	x4, .L.str.27
	add	x4, x4, :lo12:.L.str.27
	add	x21, x19, x8
	add	x1, x28, x20
	mov	x0, x28
	mov	x2, x19
	mov	x3, x21
	bl	_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
.Ltmp6:                                 // EH_LABEL
// %bb.89:
	ldr	w26, [x25, :lo12:iterations]
	adrp	x9, rdMpb
	adrp	x8, rdMpe
	str	x28, [sp, #8]                   // 8-byte Folded Spill
	cmp	w26, #1
	b.lt	.LBB5_157
// %bb.90:
	ldr	x27, [x9, :lo12:rdMpb]
	ldr	x28, [x8, :lo12:rdMpe]
	adrp	x10, rdpb
	ldr	x24, [x10, :lo12:rdpb]
	mov	w25, wzr
	sub	x8, x27, x28
	sub	x8, x8, #8
	sub	x23, x24, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x10, x9, #1
	adrp	x9, rdpe
	and	x11, x10, #0x3ffffffffffffffc
	ldr	x22, [x9, :lo12:rdpe]
	mov	w9, #32                         // =0x20
	lsl	x8, x11, #3
	stur	x10, [x29, #-8]                 // 8-byte Folded Spill
	sub	x10, x27, x24
	ccmp	x10, x9, #0, hs
	str	x11, [sp, #48]                  // 8-byte Folded Spill
	sub	x11, x27, #16
	sub	x9, x24, x8
	sub	x8, x27, x8
	cset	w20, lo
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	sub	x8, x24, #16
	stp	x9, x11, [sp, #32]              // 16-byte Folded Spill
	str	x8, [sp, #16]                   // 8-byte Folded Spill
	b	.LBB5_93
.LBB5_91:                               //   in Loop: Header=BB5_93 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w26, [x8, :lo12:iterations]
.LBB5_92:                               //   in Loop: Header=BB5_93 Depth=1
	add	w25, w25, #1
	cmp	w25, w26
	b.ge	.LBB5_110
.LBB5_93:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_96 Depth 2
                                        //     Child Loop BB5_99 Depth 2
                                        //     Child Loop BB5_103 Depth 2
                                        //       Child Loop BB5_104 Depth 3
                                        //     Child Loop BB5_108 Depth 2
	cmp	x27, x28
	b.eq	.LBB5_100
// %bb.94:                              //   in Loop: Header=BB5_93 Depth=1
	mov	x9, x24
	mov	x8, x27
	tbnz	w20, #0, .LBB5_98
// %bb.95:                              // %.preheader72
                                        //   in Loop: Header=BB5_93 Depth=1
	ldp	x9, x10, [sp, #40]              // 16-byte Folded Reload
	ldr	x8, [sp, #16]                   // 8-byte Folded Reload
.LBB5_96:                               //   Parent Loop BB5_93 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q0, [x9, #-16]
	sub	x10, x10, #4
	sub	x9, x9, #32
	stp	q1, q0, [x8, #-16]
	sub	x8, x8, #32
	cbnz	x10, .LBB5_96
// %bb.97:                              //   in Loop: Header=BB5_93 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	ldr	x9, [sp, #48]                   // 8-byte Folded Reload
	cmp	x8, x9
	ldp	x8, x9, [sp, #24]               // 16-byte Folded Reload
	b.eq	.LBB5_100
.LBB5_98:                               // %.preheader68
                                        //   in Loop: Header=BB5_93 Depth=1
	sub	x9, x9, #8
.LBB5_99:                               //   Parent Loop BB5_93 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x8, #-8]!
	cmp	x8, x28
	str	d0, [x9], #-8
	b.ne	.LBB5_99
.LBB5_100:                              //   in Loop: Header=BB5_93 Depth=1
	cmp	x23, x22
	b.eq	.LBB5_107
// %bb.101:                             // %.preheader67
                                        //   in Loop: Header=BB5_93 Depth=1
	mov	x8, #-8                         // =0xfffffffffffffff8
	mov	x9, x23
	b	.LBB5_103
.LBB5_102:                              // %._crit_edge277
                                        //   in Loop: Header=BB5_103 Depth=2
	add	x10, x24, x10
	cmp	x9, x22
	sub	x8, x8, #8
	stur	d0, [x10, #-8]
	b.eq	.LBB5_107
.LBB5_103:                              //   Parent Loop BB5_93 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_104 Depth 3
	ldr	d0, [x9, #-8]!
	mov	x10, x8
.LBB5_104:                              //   Parent Loop BB5_93 Depth=1
                                        //     Parent Loop BB5_103 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x24, x10]
	fcmp	d0, d1
	b.pl	.LBB5_102
// %bb.105:                             //   in Loop: Header=BB5_104 Depth=3
	add	x11, x24, x10
	adds	x10, x10, #8
	stur	d1, [x11, #-8]
	b.ne	.LBB5_104
// %bb.106:                             //   in Loop: Header=BB5_103 Depth=2
	cmp	x9, x22
	sub	x8, x8, #8
	stur	d0, [x24, #-8]
	b.ne	.LBB5_103
.LBB5_107:                              // %.preheader65
                                        //   in Loop: Header=BB5_93 Depth=1
	mov	x8, x23
.LBB5_108:                              //   Parent Loop BB5_93 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x22
	b.eq	.LBB5_92
// %bb.109:                             //   in Loop: Header=BB5_108 Depth=2
	mov	x9, x8
	ldr	d0, [x8]
	ldr	d1, [x9, #-8]!
	mov	x8, x9
	fcmp	d1, d0
	b.pl	.LBB5_108
	b	.LBB5_91
.LBB5_110:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w26, #1
	adrp	x24, current_test
	b.lt	.LBB5_156
// %bb.111:
	mov	w8, #15992                      // =0x3e78
	mov	w20, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	add	x23, x19, x8
	adrp	x25, iterations
	b	.LBB5_114
.LBB5_112:                              //   in Loop: Header=BB5_114 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
	ldr	w26, [x25, :lo12:iterations]
.LBB5_113:                              //   in Loop: Header=BB5_114 Depth=1
	add	w20, w20, #1
	cmp	w20, w26
	b.ge	.LBB5_123
.LBB5_114:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_117 Depth 2
                                        //       Child Loop BB5_118 Depth 3
                                        //     Child Loop BB5_121 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	mov	w8, #8                          // =0x8
	mov	x9, x23
	b	.LBB5_117
.LBB5_115:                              //   in Loop: Header=BB5_117 Depth=2
	mov	x10, x21
.LBB5_116:                              //   in Loop: Header=BB5_117 Depth=2
	cmp	x9, x19
	add	x8, x8, #8
	stur	d0, [x10, #-8]
	b.eq	.LBB5_120
.LBB5_117:                              //   Parent Loop BB5_114 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_118 Depth 3
	mov	x10, x9
	ldr	d0, [x9, #-8]!
	mov	x11, x8
.LBB5_118:                              //   Parent Loop BB5_114 Depth=1
                                        //     Parent Loop BB5_117 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x10]
	fcmp	d0, d1
	b.pl	.LBB5_116
// %bb.119:                             //   in Loop: Header=BB5_118 Depth=3
	add	x12, x10, #8
	subs	x11, x11, #8
	stur	d1, [x10, #-8]
	mov	x10, x12
	b.ne	.LBB5_118
	b	.LBB5_115
.LBB5_120:                              // %.preheader63
                                        //   in Loop: Header=BB5_114 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x23
	mov	x10, x21
.LBB5_121:                              //   Parent Loop BB5_114 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_113
// %bb.122:                             //   in Loop: Header=BB5_121 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_121
	b	.LBB5_112
.LBB5_123:
	cmp	w26, #1
	b.lt	.LBB5_157
// %bb.124:
	mov	w8, #15992                      // =0x3e78
	mov	w20, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	add	x23, x19, x8
	b	.LBB5_127
.LBB5_125:                              //   in Loop: Header=BB5_127 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
	ldr	w26, [x25, :lo12:iterations]
.LBB5_126:                              //   in Loop: Header=BB5_127 Depth=1
	add	w20, w20, #1
	cmp	w20, w26
	b.ge	.LBB5_136
.LBB5_127:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_130 Depth 2
                                        //       Child Loop BB5_131 Depth 3
                                        //     Child Loop BB5_134 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	mov	w8, #8                          // =0x8
	mov	x9, x23
	b	.LBB5_130
.LBB5_128:                              //   in Loop: Header=BB5_130 Depth=2
	mov	x10, x21
.LBB5_129:                              //   in Loop: Header=BB5_130 Depth=2
	cmp	x9, x19
	add	x8, x8, #8
	stur	d0, [x10, #-8]
	b.eq	.LBB5_133
.LBB5_130:                              //   Parent Loop BB5_127 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_131 Depth 3
	mov	x10, x9
	ldr	d0, [x9, #-8]!
	mov	x11, x8
.LBB5_131:                              //   Parent Loop BB5_127 Depth=1
                                        //     Parent Loop BB5_130 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	ldr	d1, [x10]
	fcmp	d0, d1
	b.pl	.LBB5_129
// %bb.132:                             //   in Loop: Header=BB5_131 Depth=3
	add	x12, x10, #8
	subs	x11, x11, #8
	stur	d1, [x10, #-8]
	mov	x10, x12
	b.ne	.LBB5_131
	b	.LBB5_128
.LBB5_133:                              // %.preheader61
                                        //   in Loop: Header=BB5_127 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x23
	mov	x10, x21
.LBB5_134:                              //   Parent Loop BB5_127 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_126
// %bb.135:                             //   in Loop: Header=BB5_134 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_134
	b	.LBB5_125
.LBB5_136:
	cmp	w26, #1
	b.lt	.LBB5_157
// %bb.137:
	adrp	x8, rrdMpb+8
	adrp	x9, rrdMpe+8
	adrp	x10, rrdpb+8
	ldr	x28, [x8, :lo12:rrdMpb+8]
	ldr	x25, [x9, :lo12:rrdMpe+8]
	adrp	x11, rrdpe+8
	ldr	x24, [x10, :lo12:rrdpb+8]
	ldr	x22, [x11, :lo12:rrdpe+8]
	mov	w27, wzr
	sub	x8, x25, x28
	sub	x8, x8, #8
	add	x23, x24, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x10, x9, #1
	mov	w9, #32                         // =0x20
	and	x11, x10, #0x3ffffffffffffffc
	stur	x10, [x29, #-8]                 // 8-byte Folded Spill
	sub	x10, x24, x28
	lsl	x8, x11, #3
	ccmp	x10, x9, #0, hs
	str	x11, [sp, #48]                  // 8-byte Folded Spill
	add	x11, x28, #16
	cset	w20, lo
	add	x9, x24, x8
	add	x8, x28, x8
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	add	x8, x24, #16
	stp	x9, x11, [sp, #32]              // 16-byte Folded Spill
	str	x8, [sp, #16]                   // 8-byte Folded Spill
	b	.LBB5_140
.LBB5_138:                              //   in Loop: Header=BB5_140 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w26, [x8, :lo12:iterations]
.LBB5_139:                              //   in Loop: Header=BB5_140 Depth=1
	add	w27, w27, #1
	cmp	w27, w26
	b.ge	.LBB5_204
.LBB5_140:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_143 Depth 2
                                        //     Child Loop BB5_145 Depth 2
                                        //     Child Loop BB5_149 Depth 2
                                        //       Child Loop BB5_150 Depth 3
                                        //     Child Loop BB5_154 Depth 2
	cmp	x28, x25
	b.eq	.LBB5_146
// %bb.141:                             //   in Loop: Header=BB5_140 Depth=1
	mov	x8, x24
	mov	x9, x28
	tbnz	w20, #0, .LBB5_145
// %bb.142:                             // %.preheader60
                                        //   in Loop: Header=BB5_140 Depth=1
	ldp	x9, x10, [sp, #40]              // 16-byte Folded Reload
	ldr	x8, [sp, #16]                   // 8-byte Folded Reload
.LBB5_143:                              //   Parent Loop BB5_140 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	stp	q0, q1, [x8, #-16]
	add	x8, x8, #32
	b.ne	.LBB5_143
// %bb.144:                             //   in Loop: Header=BB5_140 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	ldr	x9, [sp, #48]                   // 8-byte Folded Reload
	cmp	x8, x9
	ldp	x9, x8, [sp, #24]               // 16-byte Folded Reload
	b.eq	.LBB5_146
.LBB5_145:                              //   Parent Loop BB5_140 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x25
	str	d0, [x8], #8
	b.ne	.LBB5_145
.LBB5_146:                              //   in Loop: Header=BB5_140 Depth=1
	cmp	x23, x22
	b.eq	.LBB5_153
// %bb.147:                             // %.preheader55
                                        //   in Loop: Header=BB5_140 Depth=1
	mov	x8, xzr
	mov	x9, x23
	b	.LBB5_149
.LBB5_148:                              // %._crit_edge227
                                        //   in Loop: Header=BB5_149 Depth=2
	add	x10, x11, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x10]
	cmp	x9, x22
	b.eq	.LBB5_153
.LBB5_149:                              //   Parent Loop BB5_140 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_150 Depth 3
	ldr	d0, [x9]
	mov	x10, x8
.LBB5_150:                              //   Parent Loop BB5_140 Depth=1
                                        //     Parent Loop BB5_149 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x11, x24, x10
	ldr	d1, [x11]
	fcmp	d0, d1
	b.pl	.LBB5_148
// %bb.151:                             //   in Loop: Header=BB5_150 Depth=3
	sub	x10, x10, #8
	str	d1, [x11, #8]
	cmn	x10, #8
	b.ne	.LBB5_150
// %bb.152:                             //   in Loop: Header=BB5_149 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x24]
	cmp	x9, x22
	b.ne	.LBB5_149
.LBB5_153:                              // %.preheader53
                                        //   in Loop: Header=BB5_140 Depth=1
	mov	x8, x23
.LBB5_154:                              //   Parent Loop BB5_140 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x22
	b.eq	.LBB5_139
// %bb.155:                             //   in Loop: Header=BB5_154 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_154
	b	.LBB5_138
.LBB5_156:
	adrp	x25, iterations
.LBB5_157:
	adrp	x9, dMpb
	lsl	w8, w26, #3
	adrp	x26, dpb
	ldr	x0, [x9, :lo12:dMpb]
	adrp	x9, dMpe
	ldr	x2, [x26, :lo12:dpb]
	ldr	x1, [x9, :lo12:dMpe]
	adrp	x9, dpe
	str	w8, [x25, :lo12:iterations]
	ldr	x3, [x9, :lo12:dpe]
.Ltmp7:                                 // EH_LABEL
	movi	d0, #0000000000000000
	adrp	x4, .L.str.34
	add	x4, x4, :lo12:.L.str.34
	bl	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
.Ltmp8:                                 // EH_LABEL
	mov	w8, #16000                      // =0x3e80
// %bb.158:
.Ltmp9:                                 // EH_LABEL
	movi	d0, #0000000000000000
	adrp	x4, .L.str.35
	add	x4, x4, :lo12:.L.str.35
	add	x1, x28, x8
	mov	x0, x28
	mov	x2, x19
	mov	x3, x21
	bl	_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
.Ltmp10:                                // EH_LABEL
// %bb.159:
	ldr	w8, [x25, :lo12:iterations]
	cmp	w8, #0
	b.le	.LBB5_246
// %bb.160:
	adrp	x8, rdMpb
	adrp	x10, rdpb
	mov	w20, wzr
	ldr	x23, [x8, :lo12:rdMpb]
	adrp	x8, rdMpe
	ldr	x25, [x10, :lo12:rdpb]
	ldr	x24, [x8, :lo12:rdMpe]
	sub	x22, x25, #8
	sub	x11, x23, #16
	sub	x8, x23, x24
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	sub	x8, x8, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x10, x9, #1
	adrp	x9, rdpe
	and	x28, x10, #0x3ffffffffffffffc
	ldr	x27, [x9, :lo12:rdpe]
	mov	w9, #32                         // =0x20
	lsl	x8, x28, #3
	str	x10, [sp, #48]                  // 8-byte Folded Spill
	sub	x10, x23, x25
	ccmp	x10, x9, #0, hs
	sub	x9, x25, x8
	sub	x8, x23, x8
	cset	w26, lo
	stp	x8, x9, [sp, #32]               // 16-byte Folded Spill
	sub	x8, x25, #16
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	b	.LBB5_163
.LBB5_161:                              //   in Loop: Header=BB5_163 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB5_162:                              //   in Loop: Header=BB5_163 Depth=1
	adrp	x8, iterations
	add	w20, w20, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w20, w8
	b.ge	.LBB5_174
.LBB5_163:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_166 Depth 2
                                        //     Child Loop BB5_169 Depth 2
                                        //     Child Loop BB5_172 Depth 2
	cmp	x23, x24
	b.eq	.LBB5_170
// %bb.164:                             //   in Loop: Header=BB5_163 Depth=1
	mov	x9, x25
	mov	x8, x23
	tbnz	w26, #0, .LBB5_168
// %bb.165:                             // %.preheader45
                                        //   in Loop: Header=BB5_163 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x28
.LBB5_166:                              //   Parent Loop BB5_163 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q0, [x9, #-16]
	sub	x10, x10, #4
	sub	x9, x9, #32
	stp	q1, q0, [x8, #-16]
	sub	x8, x8, #32
	cbnz	x10, .LBB5_166
// %bb.167:                             //   in Loop: Header=BB5_163 Depth=1
	ldp	x9, x8, [sp, #40]               // 16-byte Folded Reload
	cmp	x8, x28
	ldr	x8, [sp, #32]                   // 8-byte Folded Reload
	b.eq	.LBB5_170
.LBB5_168:                              // %.preheader41
                                        //   in Loop: Header=BB5_163 Depth=1
	sub	x9, x9, #8
.LBB5_169:                              //   Parent Loop BB5_163 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x8, #-8]!
	cmp	x8, x24
	str	d0, [x9], #-8
	b.ne	.LBB5_169
.LBB5_170:                              //   in Loop: Header=BB5_163 Depth=1
	stur	x25, [x29, #-32]
	stur	x27, [x29, #-48]
.Ltmp11:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
.Ltmp12:                                // EH_LABEL
// %bb.171:                             // %.preheader39
                                        //   in Loop: Header=BB5_163 Depth=1
	mov	x8, x22
.LBB5_172:                              //   Parent Loop BB5_163 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x27
	b.eq	.LBB5_162
// %bb.173:                             //   in Loop: Header=BB5_172 Depth=2
	mov	x9, x8
	ldr	d0, [x8]
	ldr	d1, [x9, #-8]!
	mov	x8, x9
	fcmp	d1, d0
	b.pl	.LBB5_172
	b	.LBB5_161
.LBB5_174:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w8, #1
	adrp	x25, iterations
	adrp	x24, current_test
	adrp	x26, dpb
	b.lt	.LBB5_246
// %bb.175:
	mov	w8, #15992                      // =0x3e78
	mov	w23, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	add	x20, x19, x8
	b	.LBB5_178
.LBB5_176:                              //   in Loop: Header=BB5_178 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_177:                              //   in Loop: Header=BB5_178 Depth=1
	ldr	w8, [x25, :lo12:iterations]
	add	w23, w23, #1
	cmp	w23, w8
	b.ge	.LBB5_182
.LBB5_178:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_180 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x21, [x29, #-32]
	stur	x19, [x29, #-48]
.Ltmp14:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
.Ltmp15:                                // EH_LABEL
// %bb.179:                             // %.preheader37
                                        //   in Loop: Header=BB5_178 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x20
	mov	x10, x21
.LBB5_180:                              //   Parent Loop BB5_178 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_177
// %bb.181:                             //   in Loop: Header=BB5_180 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_180
	b	.LBB5_176
.LBB5_182:
	cmp	w8, #1
	b.lt	.LBB5_246
// %bb.183:                             // %.preheader36
	mov	w23, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_186
.LBB5_184:                              //   in Loop: Header=BB5_186 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_185:                              //   in Loop: Header=BB5_186 Depth=1
	ldr	w8, [x25, :lo12:iterations]
	add	w23, w23, #1
	cmp	w23, w8
	b.ge	.LBB5_190
.LBB5_186:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_188 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x21, [x29, #-32]
	stur	x19, [x29, #-48]
.Ltmp17:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
.Ltmp18:                                // EH_LABEL
// %bb.187:                             // %.preheader34
                                        //   in Loop: Header=BB5_186 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x20
	mov	x10, x21
.LBB5_188:                              //   Parent Loop BB5_186 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_185
// %bb.189:                             //   in Loop: Header=BB5_188 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_188
	b	.LBB5_184
.LBB5_190:
	cmp	w8, #0
	b.le	.LBB5_246
// %bb.191:
	adrp	x8, rrdMpb+8
	adrp	x9, rrdMpe+8
	adrp	x10, rrdpb+8
	ldr	x23, [x8, :lo12:rrdMpb+8]
	ldr	x24, [x9, :lo12:rrdMpe+8]
	ldr	x25, [x10, :lo12:rrdpb+8]
	adrp	x11, rrdpe+8
	mov	w20, wzr
	sub	x8, x24, x23
	ldr	x27, [x11, :lo12:rrdpe+8]
	add	x11, x23, #16
	sub	x8, x8, #8
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x25, #16
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x26, x25, #8
	add	x10, x9, #1
	mov	w9, #32                         // =0x20
	and	x28, x10, #0x3ffffffffffffffc
	str	x10, [sp, #48]                  // 8-byte Folded Spill
	sub	x10, x25, x23
	lsl	x8, x28, #3
	ccmp	x10, x9, #0, hs
	cset	w22, lo
	add	x9, x25, x8
	add	x8, x23, x8
	stp	x9, x11, [sp, #32]              // 16-byte Folded Spill
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	b	.LBB5_194
.LBB5_192:                              //   in Loop: Header=BB5_194 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB5_193:                              //   in Loop: Header=BB5_194 Depth=1
	adrp	x8, iterations
	add	w20, w20, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w20, w8
	b.ge	.LBB5_217
.LBB5_194:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_197 Depth 2
                                        //     Child Loop BB5_199 Depth 2
                                        //     Child Loop BB5_202 Depth 2
	cmp	x23, x24
	b.eq	.LBB5_200
// %bb.195:                             //   in Loop: Header=BB5_194 Depth=1
	mov	x8, x25
	mov	x9, x23
	tbnz	w22, #0, .LBB5_199
// %bb.196:                             // %.preheader33
                                        //   in Loop: Header=BB5_194 Depth=1
	ldr	x8, [sp, #40]                   // 8-byte Folded Reload
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x28
.LBB5_197:                              //   Parent Loop BB5_194 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	stp	q0, q1, [x8, #-16]
	add	x8, x8, #32
	b.ne	.LBB5_197
// %bb.198:                             //   in Loop: Header=BB5_194 Depth=1
	ldr	x8, [sp, #48]                   // 8-byte Folded Reload
	cmp	x8, x28
	ldp	x9, x8, [sp, #24]               // 16-byte Folded Reload
	b.eq	.LBB5_200
.LBB5_199:                              //   Parent Loop BB5_194 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x24
	str	d0, [x8], #8
	b.ne	.LBB5_199
.LBB5_200:                              //   in Loop: Header=BB5_194 Depth=1
	stur	x25, [x29, #-24]
	stur	x27, [x29, #-40]
.Ltmp20:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
.Ltmp21:                                // EH_LABEL
// %bb.201:                             // %.preheader27
                                        //   in Loop: Header=BB5_194 Depth=1
	mov	x8, x26
.LBB5_202:                              //   Parent Loop BB5_194 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x27
	b.eq	.LBB5_193
// %bb.203:                             //   in Loop: Header=BB5_202 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_202
	b	.LBB5_192
.LBB5_204:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w26, #1
	adrp	x25, iterations
	adrp	x27, current_test
	b.lt	.LBB5_157
// %bb.205:                             // %.preheader52
	mov	w20, wzr
	mov	w23, #16000                     // =0x3e80
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_208
.LBB5_206:                              //   in Loop: Header=BB5_208 Depth=1
	ldr	w1, [x27, :lo12:current_test]
	mov	x0, x22
	bl	printf
	ldr	w26, [x25, :lo12:iterations]
.LBB5_207:                              //   in Loop: Header=BB5_208 Depth=1
	add	w20, w20, #1
	cmp	w20, w26
	b.ge	.LBB5_225
.LBB5_208:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_210 Depth 2
                                        //       Child Loop BB5_211 Depth 3
                                        //     Child Loop BB5_215 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	mov	x8, xzr
	mov	w9, #8                          // =0x8
	b	.LBB5_210
.LBB5_209:                              // %._crit_edge200
                                        //   in Loop: Header=BB5_210 Depth=2
	add	x10, x11, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x10]
	cmp	x9, x23
	b.eq	.LBB5_214
.LBB5_210:                              //   Parent Loop BB5_208 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_211 Depth 3
	ldr	d0, [x19, x9]
	mov	x10, x8
.LBB5_211:                              //   Parent Loop BB5_208 Depth=1
                                        //     Parent Loop BB5_210 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x11, x19, x10
	ldr	d1, [x11]
	fcmp	d0, d1
	b.pl	.LBB5_209
// %bb.212:                             //   in Loop: Header=BB5_211 Depth=3
	sub	x10, x10, #8
	str	d1, [x11, #8]
	cmn	x10, #8
	b.ne	.LBB5_211
// %bb.213:                             //   in Loop: Header=BB5_210 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x19]
	cmp	x9, x23
	b.ne	.LBB5_210
.LBB5_214:                              // %.preheader50
                                        //   in Loop: Header=BB5_208 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
.LBB5_215:                              //   Parent Loop BB5_208 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_207
// %bb.216:                             //   in Loop: Header=BB5_215 Depth=2
	add	x9, x19, x8
	add	x8, x8, #8
	ldr	d0, [x9, #16000]
	ldr	d1, [x9, #15992]
	fcmp	d0, d1
	b.pl	.LBB5_215
	b	.LBB5_206
.LBB5_217:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w8, #1
	adrp	x25, iterations
	adrp	x24, current_test
	adrp	x26, dpb
	b.lt	.LBB5_246
// %bb.218:
	mov	w20, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_221
.LBB5_219:                              //   in Loop: Header=BB5_221 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_220:                              //   in Loop: Header=BB5_221 Depth=1
	ldr	w8, [x25, :lo12:iterations]
	add	w20, w20, #1
	cmp	w20, w8
	b.ge	.LBB5_238
.LBB5_221:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_223 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x19, [x29, #-24]
	stur	x21, [x29, #-40]
.Ltmp23:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
.Ltmp24:                                // EH_LABEL
// %bb.222:                             // %.preheader25
                                        //   in Loop: Header=BB5_221 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
.LBB5_223:                              //   Parent Loop BB5_221 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_220
// %bb.224:                             //   in Loop: Header=BB5_223 Depth=2
	add	x9, x19, x8
	add	x8, x8, #8
	ldr	d0, [x9, #16000]
	ldr	d1, [x9, #15992]
	fcmp	d0, d1
	b.pl	.LBB5_223
	b	.LBB5_219
.LBB5_225:
	cmp	w26, #1
	b.lt	.LBB5_157
// %bb.226:                             // %.preheader48
	mov	w20, wzr
	add	x23, x19, #8
	mov	w24, #16000                     // =0x3e80
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_229
.LBB5_227:                              //   in Loop: Header=BB5_229 Depth=1
	ldr	w1, [x27, :lo12:current_test]
	mov	x0, x22
	bl	printf
	ldr	w26, [x25, :lo12:iterations]
.LBB5_228:                              //   in Loop: Header=BB5_229 Depth=1
	add	w20, w20, #1
	cmp	w20, w26
	b.ge	.LBB5_157
.LBB5_229:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_231 Depth 2
                                        //       Child Loop BB5_232 Depth 3
                                        //     Child Loop BB5_236 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	mov	x8, xzr
	mov	w9, #8                          // =0x8
	b	.LBB5_231
.LBB5_230:                              // %._crit_edge
                                        //   in Loop: Header=BB5_231 Depth=2
	add	x10, x11, #8
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x10]
	cmp	x9, x24
	b.eq	.LBB5_235
.LBB5_231:                              //   Parent Loop BB5_229 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB5_232 Depth 3
	ldr	d0, [x19, x9]
	mov	x10, x8
.LBB5_232:                              //   Parent Loop BB5_229 Depth=1
                                        //     Parent Loop BB5_231 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x11, x19, x10
	ldr	d1, [x11]
	fcmp	d0, d1
	b.pl	.LBB5_230
// %bb.233:                             //   in Loop: Header=BB5_232 Depth=3
	sub	x10, x10, #8
	str	d1, [x11, #8]
	cmn	x10, #8
	b.ne	.LBB5_232
// %bb.234:                             //   in Loop: Header=BB5_231 Depth=2
	add	x9, x9, #8
	add	x8, x8, #8
	str	d0, [x19]
	cmp	x9, x24
	b.ne	.LBB5_231
.LBB5_235:                              // %.preheader46
                                        //   in Loop: Header=BB5_229 Depth=1
	mov	w8, #15992                      // =0x3e78
	mov	x9, x23
.LBB5_236:                              //   Parent Loop BB5_229 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_228
// %bb.237:                             //   in Loop: Header=BB5_236 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_236
	b	.LBB5_227
.LBB5_238:
	cmp	w8, #1
	b.lt	.LBB5_246
// %bb.239:
	mov	w20, wzr
	add	x23, x19, #8
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_242
.LBB5_240:                              //   in Loop: Header=BB5_242 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_241:                              //   in Loop: Header=BB5_242 Depth=1
	ldr	w8, [x25, :lo12:iterations]
	add	w20, w20, #1
	cmp	w20, w8
	b.ge	.LBB5_246
.LBB5_242:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_244 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x19, [x29, #-24]
	stur	x21, [x29, #-40]
.Ltmp26:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
.Ltmp27:                                // EH_LABEL
// %bb.243:                             // %.preheader22
                                        //   in Loop: Header=BB5_242 Depth=1
	mov	w8, #15992                      // =0x3e78
	mov	x9, x23
.LBB5_244:                              //   Parent Loop BB5_242 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_241
// %bb.245:                             //   in Loop: Header=BB5_244 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_244
	b	.LBB5_240
.LBB5_246:
	adrp	x8, dMpb
	ldr	x2, [x26, :lo12:dpb]
	ldr	x0, [x8, :lo12:dMpb]
	adrp	x8, dMpe
	ldr	x1, [x8, :lo12:dMpe]
	adrp	x8, dpe
	ldr	x3, [x8, :lo12:dpe]
.Ltmp29:                                // EH_LABEL
	movi	d0, #0000000000000000
	adrp	x4, .L.str.42
	add	x4, x4, :lo12:.L.str.42
	bl	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
.Ltmp30:                                // EH_LABEL
	mov	w8, #16000                      // =0x3e80
// %bb.247:
.Ltmp31:                                // EH_LABEL
	movi	d0, #0000000000000000
	adrp	x4, .L.str.43
	add	x4, x4, :lo12:.L.str.43
	add	x1, x28, x8
	mov	x0, x28
	mov	x2, x19
	mov	x3, x21
	bl	_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
.Ltmp32:                                // EH_LABEL
// %bb.248:
	ldr	w8, [x25, :lo12:iterations]
	cmp	w8, #0
	b.le	.LBB5_309
// %bb.249:
	adrp	x8, rdMpb
	adrp	x10, rdpb
	mov	w20, wzr
	ldr	x23, [x8, :lo12:rdMpb]
	adrp	x8, rdMpe
	ldr	x25, [x10, :lo12:rdpb]
	ldr	x24, [x8, :lo12:rdMpe]
	sub	x22, x25, #8
	sub	x11, x23, #16
	sub	x8, x23, x24
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	sub	x8, x8, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x10, x9, #1
	adrp	x9, rdpe
	and	x28, x10, #0x3ffffffffffffffc
	ldr	x27, [x9, :lo12:rdpe]
	mov	w9, #32                         // =0x20
	lsl	x8, x28, #3
	str	x10, [sp, #48]                  // 8-byte Folded Spill
	sub	x10, x23, x25
	ccmp	x10, x9, #0, hs
	sub	x9, x25, x8
	sub	x8, x23, x8
	cset	w26, lo
	stp	x8, x9, [sp, #32]               // 16-byte Folded Spill
	sub	x8, x25, #16
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	b	.LBB5_252
.LBB5_250:                              //   in Loop: Header=BB5_252 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB5_251:                              //   in Loop: Header=BB5_252 Depth=1
	adrp	x8, iterations
	add	w20, w20, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w20, w8
	b.ge	.LBB5_263
.LBB5_252:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_255 Depth 2
                                        //     Child Loop BB5_258 Depth 2
                                        //     Child Loop BB5_261 Depth 2
	cmp	x23, x24
	b.eq	.LBB5_259
// %bb.253:                             //   in Loop: Header=BB5_252 Depth=1
	mov	x9, x25
	mov	x8, x23
	tbnz	w26, #0, .LBB5_257
// %bb.254:                             // %.preheader21
                                        //   in Loop: Header=BB5_252 Depth=1
	ldr	x8, [sp, #24]                   // 8-byte Folded Reload
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x28
.LBB5_255:                              //   Parent Loop BB5_252 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q1, q0, [x9, #-16]
	sub	x10, x10, #4
	sub	x9, x9, #32
	stp	q1, q0, [x8, #-16]
	sub	x8, x8, #32
	cbnz	x10, .LBB5_255
// %bb.256:                             //   in Loop: Header=BB5_252 Depth=1
	ldp	x9, x8, [sp, #40]               // 16-byte Folded Reload
	cmp	x8, x28
	ldr	x8, [sp, #32]                   // 8-byte Folded Reload
	b.eq	.LBB5_259
.LBB5_257:                              // %.preheader17
                                        //   in Loop: Header=BB5_252 Depth=1
	sub	x9, x9, #8
.LBB5_258:                              //   Parent Loop BB5_252 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x8, #-8]!
	cmp	x8, x24
	str	d0, [x9], #-8
	b.ne	.LBB5_258
.LBB5_259:                              //   in Loop: Header=BB5_252 Depth=1
	stur	x25, [x29, #-32]
	stur	x27, [x29, #-48]
.Ltmp34:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_
.Ltmp35:                                // EH_LABEL
// %bb.260:                             // %.preheader15
                                        //   in Loop: Header=BB5_252 Depth=1
	mov	x8, x22
.LBB5_261:                              //   Parent Loop BB5_252 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x27
	b.eq	.LBB5_251
// %bb.262:                             //   in Loop: Header=BB5_261 Depth=2
	mov	x9, x8
	ldr	d0, [x8]
	ldr	d1, [x9, #-8]!
	mov	x8, x9
	fcmp	d1, d0
	b.pl	.LBB5_261
	b	.LBB5_250
.LBB5_263:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w8, #1
	b.lt	.LBB5_309
// %bb.264:
	mov	w8, #15992                      // =0x3e78
	mov	w23, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	add	x20, x19, x8
	adrp	x24, iterations
	adrp	x25, current_test
	b	.LBB5_267
.LBB5_265:                              //   in Loop: Header=BB5_267 Depth=1
	ldr	w1, [x25, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_266:                              //   in Loop: Header=BB5_267 Depth=1
	ldr	w8, [x24, :lo12:iterations]
	add	w23, w23, #1
	cmp	w23, w8
	b.ge	.LBB5_271
.LBB5_267:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_269 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x21, [x29, #-32]
	stur	x19, [x29, #-48]
.Ltmp37:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
.Ltmp38:                                // EH_LABEL
// %bb.268:                             // %.preheader13
                                        //   in Loop: Header=BB5_267 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x20
	mov	x10, x21
.LBB5_269:                              //   Parent Loop BB5_267 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_266
// %bb.270:                             //   in Loop: Header=BB5_269 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_269
	b	.LBB5_265
.LBB5_271:
	cmp	w8, #1
	b.lt	.LBB5_309
// %bb.272:                             // %.preheader12
	mov	w23, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_275
.LBB5_273:                              //   in Loop: Header=BB5_275 Depth=1
	ldr	w1, [x25, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_274:                              //   in Loop: Header=BB5_275 Depth=1
	ldr	w8, [x24, :lo12:iterations]
	add	w23, w23, #1
	cmp	w23, w8
	b.ge	.LBB5_279
.LBB5_275:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_277 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x21, [x29, #-32]
	stur	x19, [x29, #-48]
.Ltmp40:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
.Ltmp41:                                // EH_LABEL
// %bb.276:                             // %.preheader10
                                        //   in Loop: Header=BB5_275 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
	mov	x9, x20
	mov	x10, x21
.LBB5_277:                              //   Parent Loop BB5_275 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_274
// %bb.278:                             //   in Loop: Header=BB5_277 Depth=2
	ldr	d0, [x9, #-8]!
	add	x8, x8, #8
	ldr	d1, [x10, #-8]!
	fcmp	d0, d1
	b.pl	.LBB5_277
	b	.LBB5_273
.LBB5_279:
	cmp	w8, #0
	b.le	.LBB5_309
// %bb.280:
	adrp	x8, rrdMpb+8
	adrp	x9, rrdMpe+8
	adrp	x10, rrdpb+8
	ldr	x23, [x8, :lo12:rrdMpb+8]
	ldr	x24, [x9, :lo12:rrdMpe+8]
	ldr	x25, [x10, :lo12:rrdpb+8]
	adrp	x11, rrdpe+8
	mov	w20, wzr
	sub	x8, x24, x23
	ldr	x27, [x11, :lo12:rrdpe+8]
	add	x11, x23, #16
	sub	x8, x8, #8
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x25, #16
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x26, x25, #8
	add	x10, x9, #1
	mov	w9, #32                         // =0x20
	and	x28, x10, #0x3ffffffffffffffc
	str	x10, [sp, #48]                  // 8-byte Folded Spill
	sub	x10, x25, x23
	lsl	x8, x28, #3
	ccmp	x10, x9, #0, hs
	cset	w22, lo
	add	x9, x25, x8
	add	x8, x23, x8
	stp	x9, x11, [sp, #32]              // 16-byte Folded Spill
	str	x8, [sp, #24]                   // 8-byte Folded Spill
	b	.LBB5_283
.LBB5_281:                              //   in Loop: Header=BB5_283 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB5_282:                              //   in Loop: Header=BB5_283 Depth=1
	adrp	x8, iterations
	add	w20, w20, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w20, w8
	b.ge	.LBB5_293
.LBB5_283:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_286 Depth 2
                                        //     Child Loop BB5_288 Depth 2
                                        //     Child Loop BB5_291 Depth 2
	cmp	x23, x24
	b.eq	.LBB5_289
// %bb.284:                             //   in Loop: Header=BB5_283 Depth=1
	mov	x8, x25
	mov	x9, x23
	tbnz	w22, #0, .LBB5_288
// %bb.285:                             // %.preheader9
                                        //   in Loop: Header=BB5_283 Depth=1
	ldr	x8, [sp, #40]                   // 8-byte Folded Reload
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x28
.LBB5_286:                              //   Parent Loop BB5_283 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	stp	q0, q1, [x8, #-16]
	add	x8, x8, #32
	b.ne	.LBB5_286
// %bb.287:                             //   in Loop: Header=BB5_283 Depth=1
	ldr	x8, [sp, #48]                   // 8-byte Folded Reload
	cmp	x8, x28
	ldp	x9, x8, [sp, #24]               // 16-byte Folded Reload
	b.eq	.LBB5_289
.LBB5_288:                              //   Parent Loop BB5_283 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x24
	str	d0, [x8], #8
	b.ne	.LBB5_288
.LBB5_289:                              //   in Loop: Header=BB5_283 Depth=1
	stur	x25, [x29, #-24]
	stur	x27, [x29, #-40]
.Ltmp43:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
.Ltmp44:                                // EH_LABEL
// %bb.290:                             // %.preheader4
                                        //   in Loop: Header=BB5_283 Depth=1
	mov	x8, x26
.LBB5_291:                              //   Parent Loop BB5_283 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x27
	b.eq	.LBB5_282
// %bb.292:                             //   in Loop: Header=BB5_291 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_291
	b	.LBB5_281
.LBB5_293:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
	cmp	w8, #1
	adrp	x25, current_test
	b.lt	.LBB5_309
// %bb.294:
	mov	w20, wzr
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	adrp	x24, iterations
	b	.LBB5_297
.LBB5_295:                              //   in Loop: Header=BB5_297 Depth=1
	ldr	w1, [x25, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_296:                              //   in Loop: Header=BB5_297 Depth=1
	ldr	w8, [x24, :lo12:iterations]
	add	w20, w20, #1
	cmp	w20, w8
	b.ge	.LBB5_301
.LBB5_297:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_299 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x19, [x29, #-24]
	stur	x21, [x29, #-40]
.Ltmp46:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
.Ltmp47:                                // EH_LABEL
// %bb.298:                             // %.preheader2
                                        //   in Loop: Header=BB5_297 Depth=1
	mov	x8, #-15992                     // =0xffffffffffffc188
.LBB5_299:                              //   Parent Loop BB5_297 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_296
// %bb.300:                             //   in Loop: Header=BB5_299 Depth=2
	add	x9, x19, x8
	add	x8, x8, #8
	ldr	d0, [x9, #16000]
	ldr	d1, [x9, #15992]
	fcmp	d0, d1
	b.pl	.LBB5_299
	b	.LBB5_295
.LBB5_301:
	cmp	w8, #1
	b.lt	.LBB5_309
// %bb.302:
	mov	w20, wzr
	add	x23, x19, #8
	adrp	x22, .L.str.52
	add	x22, x22, :lo12:.L.str.52
	b	.LBB5_305
.LBB5_303:                              //   in Loop: Header=BB5_305 Depth=1
	ldr	w1, [x25, :lo12:current_test]
	mov	x0, x22
	bl	printf
.LBB5_304:                              //   in Loop: Header=BB5_305 Depth=1
	ldr	w8, [x24, :lo12:iterations]
	add	w20, w20, #1
	cmp	w20, w8
	b.ge	.LBB5_309
.LBB5_305:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB5_307 Depth 2
	mov	x0, x19
	mov	x1, x28
	mov	w2, #16000                      // =0x3e80
	bl	memcpy
	stur	x19, [x29, #-24]
	stur	x21, [x29, #-40]
.Ltmp49:                                // EH_LABEL
	sub	x0, x29, #32
	sub	x1, x29, #48
	bl	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
.Ltmp50:                                // EH_LABEL
// %bb.306:                             // %.preheader
                                        //   in Loop: Header=BB5_305 Depth=1
	mov	w8, #15992                      // =0x3e78
	mov	x9, x23
.LBB5_307:                              //   Parent Loop BB5_305 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cbz	x8, .LBB5_304
// %bb.308:                             //   in Loop: Header=BB5_307 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	sub	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB5_307
	b	.LBB5_303
.LBB5_309:
	mov	x0, x28
	mov	w1, #16000                      // =0x3e80
	bl	_ZdlPvm
	mov	x0, x19
	mov	w1, #16000                      // =0x3e80
	bl	_ZdlPvm
	mov	w0, wzr
	.cfi_def_cfa wsp, 208
	ldp	x20, x19, [sp, #192]            // 16-byte Folded Reload
	ldr	d8, [sp, #96]                   // 8-byte Folded Reload
	ldp	x22, x21, [sp, #176]            // 16-byte Folded Reload
	ldp	x24, x23, [sp, #160]            // 16-byte Folded Reload
	ldp	x26, x25, [sp, #144]            // 16-byte Folded Reload
	ldp	x28, x27, [sp, #128]            // 16-byte Folded Reload
	ldp	x29, x30, [sp, #112]            // 16-byte Folded Reload
	add	sp, sp, #208
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
.LBB5_310:
	.cfi_restore_state
.Ltmp2:                                 // EH_LABEL
	mov	x21, x0
	mov	x0, x19
	mov	w1, #16000                      // =0x3e80
	bl	_ZdlPvm
	mov	x0, x21
	bl	_Unwind_Resume
.LBB5_311:
.Ltmp51:                                // EH_LABEL
	b	.LBB5_325
.LBB5_312:
.Ltmp28:                                // EH_LABEL
	b	.LBB5_325
.LBB5_313:
.Ltmp48:                                // EH_LABEL
	b	.LBB5_325
.LBB5_314:
.Ltmp25:                                // EH_LABEL
	b	.LBB5_325
.LBB5_315:
.Ltmp45:                                // EH_LABEL
	b	.LBB5_324
.LBB5_316:
.Ltmp22:                                // EH_LABEL
	b	.LBB5_324
.LBB5_317:
.Ltmp33:                                // EH_LABEL
	b	.LBB5_325
.LBB5_318:
.Ltmp42:                                // EH_LABEL
	b	.LBB5_325
.LBB5_319:
.Ltmp19:                                // EH_LABEL
	b	.LBB5_325
.LBB5_320:
.Ltmp39:                                // EH_LABEL
	b	.LBB5_325
.LBB5_321:
.Ltmp16:                                // EH_LABEL
	b	.LBB5_325
.LBB5_322:
.Ltmp36:                                // EH_LABEL
	b	.LBB5_324
.LBB5_323:
.Ltmp13:                                // EH_LABEL
.LBB5_324:
	ldr	x28, [sp, #8]                   // 8-byte Folded Reload
.LBB5_325:
	mov	x21, x0
	mov	x0, x28
	mov	w1, #16000                      // =0x3e80
	bl	_ZdlPvm
	mov	x0, x19
	mov	w1, #16000                      // =0x3e80
	bl	_ZdlPvm
	mov	x0, x21
	bl	_Unwind_Resume
.Lfunc_end5:
	.size	main, .Lfunc_end5-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2, 0x0
GCC_except_table5:
.Lexception0:
	.byte	255                             // @LPStart Encoding = omit
	.byte	255                             // @TType Encoding = omit
	.byte	1                               // Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    // >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           //   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           // >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  //   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           //     jumps to .Ltmp2
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp1-.Lfunc_begin0           // >> Call Site 3 <<
	.uleb128 .Ltmp3-.Ltmp1                  //   Call between .Ltmp1 and .Ltmp3
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp3-.Lfunc_begin0           // >> Call Site 4 <<
	.uleb128 .Ltmp6-.Ltmp3                  //   Call between .Ltmp3 and .Ltmp6
	.uleb128 .Ltmp33-.Lfunc_begin0          //     jumps to .Ltmp33
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp6-.Lfunc_begin0           // >> Call Site 5 <<
	.uleb128 .Ltmp7-.Ltmp6                  //   Call between .Ltmp6 and .Ltmp7
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp7-.Lfunc_begin0           // >> Call Site 6 <<
	.uleb128 .Ltmp10-.Ltmp7                 //   Call between .Ltmp7 and .Ltmp10
	.uleb128 .Ltmp33-.Lfunc_begin0          //     jumps to .Ltmp33
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp11-.Lfunc_begin0          // >> Call Site 7 <<
	.uleb128 .Ltmp12-.Ltmp11                //   Call between .Ltmp11 and .Ltmp12
	.uleb128 .Ltmp13-.Lfunc_begin0          //     jumps to .Ltmp13
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp12-.Lfunc_begin0          // >> Call Site 8 <<
	.uleb128 .Ltmp14-.Ltmp12                //   Call between .Ltmp12 and .Ltmp14
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp14-.Lfunc_begin0          // >> Call Site 9 <<
	.uleb128 .Ltmp15-.Ltmp14                //   Call between .Ltmp14 and .Ltmp15
	.uleb128 .Ltmp16-.Lfunc_begin0          //     jumps to .Ltmp16
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp15-.Lfunc_begin0          // >> Call Site 10 <<
	.uleb128 .Ltmp17-.Ltmp15                //   Call between .Ltmp15 and .Ltmp17
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp17-.Lfunc_begin0          // >> Call Site 11 <<
	.uleb128 .Ltmp18-.Ltmp17                //   Call between .Ltmp17 and .Ltmp18
	.uleb128 .Ltmp19-.Lfunc_begin0          //     jumps to .Ltmp19
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp20-.Lfunc_begin0          // >> Call Site 12 <<
	.uleb128 .Ltmp21-.Ltmp20                //   Call between .Ltmp20 and .Ltmp21
	.uleb128 .Ltmp22-.Lfunc_begin0          //     jumps to .Ltmp22
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp21-.Lfunc_begin0          // >> Call Site 13 <<
	.uleb128 .Ltmp23-.Ltmp21                //   Call between .Ltmp21 and .Ltmp23
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp23-.Lfunc_begin0          // >> Call Site 14 <<
	.uleb128 .Ltmp24-.Ltmp23                //   Call between .Ltmp23 and .Ltmp24
	.uleb128 .Ltmp25-.Lfunc_begin0          //     jumps to .Ltmp25
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp24-.Lfunc_begin0          // >> Call Site 15 <<
	.uleb128 .Ltmp26-.Ltmp24                //   Call between .Ltmp24 and .Ltmp26
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp26-.Lfunc_begin0          // >> Call Site 16 <<
	.uleb128 .Ltmp27-.Ltmp26                //   Call between .Ltmp26 and .Ltmp27
	.uleb128 .Ltmp28-.Lfunc_begin0          //     jumps to .Ltmp28
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp29-.Lfunc_begin0          // >> Call Site 17 <<
	.uleb128 .Ltmp32-.Ltmp29                //   Call between .Ltmp29 and .Ltmp32
	.uleb128 .Ltmp33-.Lfunc_begin0          //     jumps to .Ltmp33
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp34-.Lfunc_begin0          // >> Call Site 18 <<
	.uleb128 .Ltmp35-.Ltmp34                //   Call between .Ltmp34 and .Ltmp35
	.uleb128 .Ltmp36-.Lfunc_begin0          //     jumps to .Ltmp36
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp35-.Lfunc_begin0          // >> Call Site 19 <<
	.uleb128 .Ltmp37-.Ltmp35                //   Call between .Ltmp35 and .Ltmp37
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp37-.Lfunc_begin0          // >> Call Site 20 <<
	.uleb128 .Ltmp38-.Ltmp37                //   Call between .Ltmp37 and .Ltmp38
	.uleb128 .Ltmp39-.Lfunc_begin0          //     jumps to .Ltmp39
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp38-.Lfunc_begin0          // >> Call Site 21 <<
	.uleb128 .Ltmp40-.Ltmp38                //   Call between .Ltmp38 and .Ltmp40
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp40-.Lfunc_begin0          // >> Call Site 22 <<
	.uleb128 .Ltmp41-.Ltmp40                //   Call between .Ltmp40 and .Ltmp41
	.uleb128 .Ltmp42-.Lfunc_begin0          //     jumps to .Ltmp42
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp43-.Lfunc_begin0          // >> Call Site 23 <<
	.uleb128 .Ltmp44-.Ltmp43                //   Call between .Ltmp43 and .Ltmp44
	.uleb128 .Ltmp45-.Lfunc_begin0          //     jumps to .Ltmp45
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp44-.Lfunc_begin0          // >> Call Site 24 <<
	.uleb128 .Ltmp46-.Ltmp44                //   Call between .Ltmp44 and .Ltmp46
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp46-.Lfunc_begin0          // >> Call Site 25 <<
	.uleb128 .Ltmp47-.Ltmp46                //   Call between .Ltmp46 and .Ltmp47
	.uleb128 .Ltmp48-.Lfunc_begin0          //     jumps to .Ltmp48
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp47-.Lfunc_begin0          // >> Call Site 26 <<
	.uleb128 .Ltmp49-.Ltmp47                //   Call between .Ltmp47 and .Ltmp49
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp49-.Lfunc_begin0          // >> Call Site 27 <<
	.uleb128 .Ltmp50-.Ltmp49                //   Call between .Ltmp49 and .Ltmp50
	.uleb128 .Ltmp51-.Lfunc_begin0          //     jumps to .Ltmp51
	.byte	0                               //   On action: cleanup
	.uleb128 .Ltmp50-.Lfunc_begin0          // >> Call Site 28 <<
	.uleb128 .Lfunc_end5-.Ltmp50            //   Call between .Ltmp50 and .Lfunc_end5
	.byte	0                               //     has no landing pad
	.byte	0                               //   On action: cleanup
.Lcst_end0:
	.p2align	2, 0x0
                                        // -- End function
	.section	.text._Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc: // @_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB6_51
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	add	x25, x2, #8
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x25, x3
	b.eq	.LBB6_20
// %bb.2:
	cmp	x22, x21
	b.eq	.LBB6_32
// %bb.3:
	sub	x9, x21, x22
	mov	w26, wzr
	add	x13, x22, #16
	sub	x9, x9, #8
	add	x12, x20, #16
	lsr	x10, x9, #3
	cmp	x9, #24
	stp	x12, x13, [sp, #8]              // 16-byte Folded Spill
	add	x11, x10, #1
	sub	x10, x20, x22
	and	x28, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	mov	w11, #32                        // =0x20
	lsl	x9, x28, #3
	ccmp	x10, x11, #0, hs
	cset	w27, lo
	add	x24, x20, x9
	add	x23, x22, x9
	b	.LBB6_6
.LBB6_4:                                //   in Loop: Header=BB6_6 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB6_5:                                //   in Loop: Header=BB6_6 Depth=1
	add	w26, w26, #1
	cmp	w26, w8
	b.ge	.LBB6_50
.LBB6_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_8 Depth 2
                                        //     Child Loop BB6_10 Depth 2
                                        //     Child Loop BB6_13 Depth 2
                                        //       Child Loop BB6_14 Depth 3
                                        //     Child Loop BB6_18 Depth 2
	mov	x9, x20
	mov	x10, x22
	tbnz	w27, #0, .LBB6_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB6_6 Depth=1
	ldp	x9, x10, [sp, #8]               // 16-byte Folded Reload
	mov	x11, x28
.LBB6_8:                                //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	subs	x11, x11, #4
	add	x10, x10, #32
	stp	q0, q1, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB6_8
// %bb.9:                               //   in Loop: Header=BB6_6 Depth=1
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x23
	cmp	x9, x28
	mov	x9, x24
	b.eq	.LBB6_11
.LBB6_10:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x10], #8
	cmp	x10, x21
	str	d0, [x9], #8
	b.ne	.LBB6_10
.LBB6_11:                               // %.preheader14
                                        //   in Loop: Header=BB6_6 Depth=1
	mov	x9, xzr
	mov	x10, x25
	b	.LBB6_13
.LBB6_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB6_13 Depth=2
	add	x11, x12, #8
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x11]
	cmp	x10, x19
	b.eq	.LBB6_17
.LBB6_13:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_14 Depth 3
	ldr	d0, [x10]
	mov	x11, x9
.LBB6_14:                               //   Parent Loop BB6_6 Depth=1
                                        //     Parent Loop BB6_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x20, x11
	ldr	d1, [x12]
	fcmp	d0, d1
	b.pl	.LBB6_12
// %bb.15:                              //   in Loop: Header=BB6_14 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB6_14
// %bb.16:                              //   in Loop: Header=BB6_13 Depth=2
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x20]
	cmp	x10, x19
	b.ne	.LBB6_13
.LBB6_17:                               // %.preheader12
                                        //   in Loop: Header=BB6_6 Depth=1
	mov	x9, x25
.LBB6_18:                               //   Parent Loop BB6_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB6_5
// %bb.19:                              //   in Loop: Header=BB6_18 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB6_18
	b	.LBB6_4
.LBB6_20:
	cmp	x22, x21
	b.eq	.LBB6_44
// %bb.21:
	sub	x9, x21, x22
	mov	w24, wzr
	add	x13, x22, #16
	sub	x9, x9, #8
	add	x12, x20, #16
	lsr	x10, x9, #3
	cmp	x9, #24
	stp	x12, x13, [sp, #8]              // 16-byte Folded Spill
	add	x11, x10, #1
	sub	x10, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	mov	w11, #32                        // =0x20
	lsl	x9, x27, #3
	ccmp	x10, x11, #0, hs
	cset	w26, lo
	add	x28, x20, x9
	add	x23, x22, x9
	b	.LBB6_24
.LBB6_22:                               //   in Loop: Header=BB6_24 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB6_23:                               //   in Loop: Header=BB6_24 Depth=1
	add	w24, w24, #1
	cmp	w24, w8
	b.ge	.LBB6_50
.LBB6_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_26 Depth 2
                                        //     Child Loop BB6_28 Depth 2
                                        //     Child Loop BB6_30 Depth 2
	mov	x9, x20
	mov	x10, x22
	tbnz	w26, #0, .LBB6_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB6_24 Depth=1
	ldp	x9, x10, [sp, #8]               // 16-byte Folded Reload
	mov	x11, x27
.LBB6_26:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x10, #-16]
	subs	x11, x11, #4
	add	x10, x10, #32
	stp	q0, q1, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB6_26
// %bb.27:                              //   in Loop: Header=BB6_24 Depth=1
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x23
	cmp	x9, x27
	mov	x9, x28
	b.eq	.LBB6_29
.LBB6_28:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x10], #8
	cmp	x10, x21
	str	d0, [x9], #8
	b.ne	.LBB6_28
.LBB6_29:                               // %.preheader2
                                        //   in Loop: Header=BB6_24 Depth=1
	mov	x9, x25
.LBB6_30:                               //   Parent Loop BB6_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB6_23
// %bb.31:                              //   in Loop: Header=BB6_30 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB6_30
	b	.LBB6_22
.LBB6_32:                               // %.preheader10
	mov	w22, wzr
	adrp	x23, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB6_35
.LBB6_33:                               //   in Loop: Header=BB6_35 Depth=1
	ldr	w1, [x23, :lo12:current_test]
	mov	x0, x21
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB6_34:                               //   in Loop: Header=BB6_35 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB6_50
.LBB6_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_37 Depth 2
                                        //       Child Loop BB6_38 Depth 3
                                        //     Child Loop BB6_42 Depth 2
	mov	x9, xzr
	mov	x10, x25
	b	.LBB6_37
.LBB6_36:                               // %._crit_edge
                                        //   in Loop: Header=BB6_37 Depth=2
	add	x11, x12, #8
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x11]
	cmp	x10, x19
	b.eq	.LBB6_41
.LBB6_37:                               //   Parent Loop BB6_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB6_38 Depth 3
	ldr	d0, [x10]
	mov	x11, x9
.LBB6_38:                               //   Parent Loop BB6_35 Depth=1
                                        //     Parent Loop BB6_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x20, x11
	ldr	d1, [x12]
	fcmp	d0, d1
	b.pl	.LBB6_36
// %bb.39:                              //   in Loop: Header=BB6_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB6_38
// %bb.40:                              //   in Loop: Header=BB6_37 Depth=2
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x20]
	cmp	x10, x19
	b.ne	.LBB6_37
.LBB6_41:                               // %.preheader8
                                        //   in Loop: Header=BB6_35 Depth=1
	mov	x9, x25
.LBB6_42:                               //   Parent Loop BB6_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB6_34
// %bb.43:                              //   in Loop: Header=BB6_42 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB6_42
	b	.LBB6_33
.LBB6_44:                               // %.preheader
	mov	w21, wzr
	adrp	x22, current_test
	adrp	x20, .L.str.52
	add	x20, x20, :lo12:.L.str.52
	b	.LBB6_47
.LBB6_45:                               //   in Loop: Header=BB6_47 Depth=1
	ldr	w1, [x22, :lo12:current_test]
	mov	x0, x20
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB6_46:                               //   in Loop: Header=BB6_47 Depth=1
	add	w21, w21, #1
	cmp	w21, w8
	b.ge	.LBB6_50
.LBB6_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB6_48 Depth 2
	mov	x9, x25
.LBB6_48:                               //   Parent Loop BB6_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB6_46
// %bb.49:                              //   in Loop: Header=BB6_48 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB6_48
	b	.LBB6_45
.LBB6_50:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB6_51:
	ret
.Lfunc_end6:
	.size	_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end6-_Z19test_insertion_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,"axG",@progbits,_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,comdat
	.weak	_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc // -- Begin function _Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.p2align	2
	.type	_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,@function
_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc: // @_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB7_51
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	add	x25, x2, #8
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x25, x3
	b.eq	.LBB7_20
// %bb.2:
	cmp	x22, x21
	b.eq	.LBB7_32
// %bb.3:
	sub	x9, x21, x22
	mov	w26, wzr
	add	x13, x20, #16
	sub	x9, x9, #8
	add	x12, x22, #16
	lsr	x10, x9, #3
	cmp	x9, #24
	stp	x12, x13, [sp, #8]              // 16-byte Folded Spill
	add	x11, x10, #1
	sub	x10, x20, x22
	and	x28, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	mov	w11, #32                        // =0x20
	lsl	x9, x28, #3
	ccmp	x10, x11, #0, hs
	cset	w27, lo
	add	x24, x22, x9
	add	x23, x20, x9
	b	.LBB7_6
.LBB7_4:                                //   in Loop: Header=BB7_6 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB7_5:                                //   in Loop: Header=BB7_6 Depth=1
	add	w26, w26, #1
	cmp	w26, w8
	b.ge	.LBB7_50
.LBB7_6:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_8 Depth 2
                                        //     Child Loop BB7_10 Depth 2
                                        //     Child Loop BB7_13 Depth 2
                                        //       Child Loop BB7_14 Depth 3
                                        //     Child Loop BB7_18 Depth 2
	mov	x9, x22
	mov	x10, x20
	tbnz	w27, #0, .LBB7_10
// %bb.7:                               // %.preheader18
                                        //   in Loop: Header=BB7_6 Depth=1
	ldp	x9, x10, [sp, #8]               // 16-byte Folded Reload
	mov	x11, x28
.LBB7_8:                                //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x11, x11, #4
	add	x9, x9, #32
	stp	q0, q1, [x10, #-16]
	add	x10, x10, #32
	b.ne	.LBB7_8
// %bb.9:                               //   in Loop: Header=BB7_6 Depth=1
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x23
	cmp	x9, x28
	mov	x9, x24
	b.eq	.LBB7_11
.LBB7_10:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x21
	str	d0, [x10], #8
	b.ne	.LBB7_10
.LBB7_11:                               // %.preheader14
                                        //   in Loop: Header=BB7_6 Depth=1
	mov	x9, xzr
	mov	x10, x25
	b	.LBB7_13
.LBB7_12:                               // %._crit_edge55
                                        //   in Loop: Header=BB7_13 Depth=2
	add	x11, x12, #8
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x11]
	cmp	x10, x19
	b.eq	.LBB7_17
.LBB7_13:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB7_14 Depth 3
	ldr	d0, [x10]
	mov	x11, x9
.LBB7_14:                               //   Parent Loop BB7_6 Depth=1
                                        //     Parent Loop BB7_13 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x20, x11
	ldr	d1, [x12]
	fcmp	d0, d1
	b.pl	.LBB7_12
// %bb.15:                              //   in Loop: Header=BB7_14 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB7_14
// %bb.16:                              //   in Loop: Header=BB7_13 Depth=2
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x20]
	cmp	x10, x19
	b.ne	.LBB7_13
.LBB7_17:                               // %.preheader12
                                        //   in Loop: Header=BB7_6 Depth=1
	mov	x9, x25
.LBB7_18:                               //   Parent Loop BB7_6 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB7_5
// %bb.19:                              //   in Loop: Header=BB7_18 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB7_18
	b	.LBB7_4
.LBB7_20:
	cmp	x22, x21
	b.eq	.LBB7_44
// %bb.21:
	sub	x9, x21, x22
	mov	w24, wzr
	add	x13, x20, #16
	sub	x9, x9, #8
	add	x12, x22, #16
	lsr	x10, x9, #3
	cmp	x9, #24
	stp	x12, x13, [sp, #8]              // 16-byte Folded Spill
	add	x11, x10, #1
	sub	x10, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	mov	w11, #32                        // =0x20
	lsl	x9, x27, #3
	ccmp	x10, x11, #0, hs
	cset	w26, lo
	add	x28, x22, x9
	add	x23, x20, x9
	b	.LBB7_24
.LBB7_22:                               //   in Loop: Header=BB7_24 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB7_23:                               //   in Loop: Header=BB7_24 Depth=1
	add	w24, w24, #1
	cmp	w24, w8
	b.ge	.LBB7_50
.LBB7_24:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_26 Depth 2
                                        //     Child Loop BB7_28 Depth 2
                                        //     Child Loop BB7_30 Depth 2
	mov	x9, x22
	mov	x10, x20
	tbnz	w26, #0, .LBB7_28
// %bb.25:                              // %.preheader6
                                        //   in Loop: Header=BB7_24 Depth=1
	ldp	x9, x10, [sp, #8]               // 16-byte Folded Reload
	mov	x11, x27
.LBB7_26:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x11, x11, #4
	add	x9, x9, #32
	stp	q0, q1, [x10, #-16]
	add	x10, x10, #32
	b.ne	.LBB7_26
// %bb.27:                              //   in Loop: Header=BB7_24 Depth=1
	ldur	x9, [x29, #-8]                  // 8-byte Folded Reload
	mov	x10, x23
	cmp	x9, x27
	mov	x9, x28
	b.eq	.LBB7_29
.LBB7_28:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x21
	str	d0, [x10], #8
	b.ne	.LBB7_28
.LBB7_29:                               // %.preheader2
                                        //   in Loop: Header=BB7_24 Depth=1
	mov	x9, x25
.LBB7_30:                               //   Parent Loop BB7_24 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB7_23
// %bb.31:                              //   in Loop: Header=BB7_30 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB7_30
	b	.LBB7_22
.LBB7_32:                               // %.preheader10
	mov	w22, wzr
	adrp	x23, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB7_35
.LBB7_33:                               //   in Loop: Header=BB7_35 Depth=1
	ldr	w1, [x23, :lo12:current_test]
	mov	x0, x21
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB7_34:                               //   in Loop: Header=BB7_35 Depth=1
	add	w22, w22, #1
	cmp	w22, w8
	b.ge	.LBB7_50
.LBB7_35:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_37 Depth 2
                                        //       Child Loop BB7_38 Depth 3
                                        //     Child Loop BB7_42 Depth 2
	mov	x9, xzr
	mov	x10, x25
	b	.LBB7_37
.LBB7_36:                               // %._crit_edge
                                        //   in Loop: Header=BB7_37 Depth=2
	add	x11, x12, #8
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x11]
	cmp	x10, x19
	b.eq	.LBB7_41
.LBB7_37:                               //   Parent Loop BB7_35 Depth=1
                                        // =>  This Loop Header: Depth=2
                                        //       Child Loop BB7_38 Depth 3
	ldr	d0, [x10]
	mov	x11, x9
.LBB7_38:                               //   Parent Loop BB7_35 Depth=1
                                        //     Parent Loop BB7_37 Depth=2
                                        // =>    This Inner Loop Header: Depth=3
	add	x12, x20, x11
	ldr	d1, [x12]
	fcmp	d0, d1
	b.pl	.LBB7_36
// %bb.39:                              //   in Loop: Header=BB7_38 Depth=3
	sub	x11, x11, #8
	str	d1, [x12, #8]
	cmn	x11, #8
	b.ne	.LBB7_38
// %bb.40:                              //   in Loop: Header=BB7_37 Depth=2
	add	x10, x10, #8
	add	x9, x9, #8
	str	d0, [x20]
	cmp	x10, x19
	b.ne	.LBB7_37
.LBB7_41:                               // %.preheader8
                                        //   in Loop: Header=BB7_35 Depth=1
	mov	x9, x25
.LBB7_42:                               //   Parent Loop BB7_35 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB7_34
// %bb.43:                              //   in Loop: Header=BB7_42 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB7_42
	b	.LBB7_33
.LBB7_44:                               // %.preheader
	mov	w21, wzr
	adrp	x22, current_test
	adrp	x20, .L.str.52
	add	x20, x20, :lo12:.L.str.52
	b	.LBB7_47
.LBB7_45:                               //   in Loop: Header=BB7_47 Depth=1
	ldr	w1, [x22, :lo12:current_test]
	mov	x0, x20
	bl	printf
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
.LBB7_46:                               //   in Loop: Header=BB7_47 Depth=1
	add	w21, w21, #1
	cmp	w21, w8
	b.ge	.LBB7_50
.LBB7_47:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB7_48 Depth 2
	mov	x9, x25
.LBB7_48:                               //   Parent Loop BB7_47 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x9, x19
	b.eq	.LBB7_46
// %bb.49:                              //   in Loop: Header=BB7_48 Depth=2
	ldp	d1, d0, [x9, #-8]
	add	x9, x9, #8
	fcmp	d0, d1
	b.pl	.LBB7_48
	b	.LBB7_45
.LBB7_50:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB7_51:
	ret
.Lfunc_end7:
	.size	_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc, .Lfunc_end7-_Z19test_insertion_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc: // @_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB8_20
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x0, x1
	b.eq	.LBB8_13
// %bb.2:
	sub	x8, x21, x22
	mov	w10, #32                        // =0x20
	mov	w25, wzr
	sub	x8, x8, #8
	add	x26, x20, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x11, x9, #1
	sub	x9, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x22, #16
	lsl	x8, x27, #3
	ccmp	x9, x10, #0, hs
	str	x11, [sp, #16]                  // 8-byte Folded Spill
	add	x11, x20, #16
	cset	w28, lo
	add	x23, x20, x8
	add	x24, x22, x8
	str	x11, [sp, #8]                   // 8-byte Folded Spill
	b	.LBB8_5
.LBB8_3:                                //   in Loop: Header=BB8_5 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB8_4:                                //   in Loop: Header=BB8_5 Depth=1
	adrp	x8, iterations
	add	w25, w25, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w25, w8
	b.ge	.LBB8_19
.LBB8_5:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_7 Depth 2
                                        //     Child Loop BB8_9 Depth 2
                                        //     Child Loop BB8_11 Depth 2
	mov	x8, x20
	mov	x9, x22
	tbnz	w28, #0, .LBB8_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB8_5 Depth=1
	ldp	x8, x9, [sp, #8]                // 16-byte Folded Reload
	mov	x10, x27
.LBB8_7:                                //   Parent Loop BB8_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	stp	q0, q1, [x8, #-16]
	add	x8, x8, #32
	b.ne	.LBB8_7
// %bb.8:                               //   in Loop: Header=BB8_5 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	mov	x9, x24
	cmp	x8, x27
	mov	x8, x23
	b.eq	.LBB8_10
.LBB8_9:                                //   Parent Loop BB8_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x21
	str	d0, [x8], #8
	b.ne	.LBB8_9
.LBB8_10:                               //   in Loop: Header=BB8_5 Depth=1
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
	mov	x8, x26
.LBB8_11:                               //   Parent Loop BB8_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB8_4
// %bb.12:                              //   in Loop: Header=BB8_11 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB8_11
	b	.LBB8_3
.LBB8_13:                               // %.preheader
	mov	w22, wzr
	add	x23, x20, #8
	adrp	x24, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB8_16
.LBB8_14:                               //   in Loop: Header=BB8_16 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x21
	bl	printf
.LBB8_15:                               //   in Loop: Header=BB8_16 Depth=1
	adrp	x8, iterations
	add	w22, w22, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w22, w8
	b.ge	.LBB8_19
.LBB8_16:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB8_17 Depth 2
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
	mov	x8, x23
.LBB8_17:                               //   Parent Loop BB8_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB8_15
// %bb.18:                              //   in Loop: Header=BB8_17 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB8_17
	b	.LBB8_14
.LBB8_19:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB8_20:
	ret
.Lfunc_end8:
	.size	_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end8-_Z14test_quicksortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,"axG",@progbits,_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,comdat
	.weak	_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc // -- Begin function _Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.p2align	2
	.type	_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,@function
_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc: // @_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB9_20
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x0, x1
	b.eq	.LBB9_13
// %bb.2:
	sub	x8, x21, x22
	mov	w10, #32                        // =0x20
	mov	w25, wzr
	sub	x8, x8, #8
	add	x26, x20, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x11, x9, #1
	sub	x9, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x20, #16
	lsl	x8, x27, #3
	ccmp	x9, x10, #0, hs
	str	x11, [sp, #16]                  // 8-byte Folded Spill
	add	x11, x22, #16
	cset	w28, lo
	add	x23, x22, x8
	add	x24, x20, x8
	str	x11, [sp, #8]                   // 8-byte Folded Spill
	b	.LBB9_5
.LBB9_3:                                //   in Loop: Header=BB9_5 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB9_4:                                //   in Loop: Header=BB9_5 Depth=1
	adrp	x8, iterations
	add	w25, w25, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w25, w8
	b.ge	.LBB9_19
.LBB9_5:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_7 Depth 2
                                        //     Child Loop BB9_9 Depth 2
                                        //     Child Loop BB9_11 Depth 2
	mov	x8, x22
	mov	x9, x20
	tbnz	w28, #0, .LBB9_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB9_5 Depth=1
	ldp	x8, x9, [sp, #8]                // 16-byte Folded Reload
	mov	x10, x27
.LBB9_7:                                //   Parent Loop BB9_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x8, #-16]
	subs	x10, x10, #4
	add	x8, x8, #32
	stp	q0, q1, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB9_7
// %bb.8:                               //   in Loop: Header=BB9_5 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	mov	x9, x24
	cmp	x8, x27
	mov	x8, x23
	b.eq	.LBB9_10
.LBB9_9:                                //   Parent Loop BB9_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x8], #8
	cmp	x8, x21
	str	d0, [x9], #8
	b.ne	.LBB9_9
.LBB9_10:                               //   in Loop: Header=BB9_5 Depth=1
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	mov	x8, x26
.LBB9_11:                               //   Parent Loop BB9_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB9_4
// %bb.12:                              //   in Loop: Header=BB9_11 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB9_11
	b	.LBB9_3
.LBB9_13:                               // %.preheader
	mov	w22, wzr
	add	x23, x20, #8
	adrp	x24, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB9_16
.LBB9_14:                               //   in Loop: Header=BB9_16 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x21
	bl	printf
.LBB9_15:                               //   in Loop: Header=BB9_16 Depth=1
	adrp	x8, iterations
	add	w22, w22, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w22, w8
	b.ge	.LBB9_19
.LBB9_16:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB9_17 Depth 2
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	mov	x8, x23
.LBB9_17:                               //   Parent Loop BB9_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB9_15
// %bb.18:                              //   in Loop: Header=BB9_17 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB9_17
	b	.LBB9_14
.LBB9_19:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB9_20:
	ret
.Lfunc_end9:
	.size	_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc, .Lfunc_end9-_Z14test_quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,"axG",@progbits,_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,comdat
	.weak	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc // -- Begin function _Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc,@function
_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc: // @_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB10_20
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x0, x1
	b.eq	.LBB10_13
// %bb.2:
	sub	x8, x21, x22
	mov	w10, #32                        // =0x20
	mov	w25, wzr
	sub	x8, x8, #8
	add	x26, x20, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x11, x9, #1
	sub	x9, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x22, #16
	lsl	x8, x27, #3
	ccmp	x9, x10, #0, hs
	str	x11, [sp, #16]                  // 8-byte Folded Spill
	add	x11, x20, #16
	cset	w28, lo
	add	x23, x20, x8
	add	x24, x22, x8
	str	x11, [sp, #8]                   // 8-byte Folded Spill
	b	.LBB10_5
.LBB10_3:                               //   in Loop: Header=BB10_5 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB10_4:                               //   in Loop: Header=BB10_5 Depth=1
	adrp	x8, iterations
	add	w25, w25, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w25, w8
	b.ge	.LBB10_19
.LBB10_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_7 Depth 2
                                        //     Child Loop BB10_9 Depth 2
                                        //     Child Loop BB10_11 Depth 2
	mov	x8, x20
	mov	x9, x22
	tbnz	w28, #0, .LBB10_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB10_5 Depth=1
	ldp	x8, x9, [sp, #8]                // 16-byte Folded Reload
	mov	x10, x27
.LBB10_7:                               //   Parent Loop BB10_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x9, #-16]
	subs	x10, x10, #4
	add	x9, x9, #32
	stp	q0, q1, [x8, #-16]
	add	x8, x8, #32
	b.ne	.LBB10_7
// %bb.8:                               //   in Loop: Header=BB10_5 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	mov	x9, x24
	cmp	x8, x27
	mov	x8, x23
	b.eq	.LBB10_10
.LBB10_9:                               //   Parent Loop BB10_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x9], #8
	cmp	x9, x21
	str	d0, [x8], #8
	b.ne	.LBB10_9
.LBB10_10:                              //   in Loop: Header=BB10_5 Depth=1
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark8heapsortIPddEEvT_S2_
	mov	x8, x26
.LBB10_11:                              //   Parent Loop BB10_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB10_4
// %bb.12:                              //   in Loop: Header=BB10_11 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_11
	b	.LBB10_3
.LBB10_13:                              // %.preheader
	mov	w22, wzr
	add	x23, x20, #8
	adrp	x24, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB10_16
.LBB10_14:                              //   in Loop: Header=BB10_16 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x21
	bl	printf
.LBB10_15:                              //   in Loop: Header=BB10_16 Depth=1
	adrp	x8, iterations
	add	w22, w22, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w22, w8
	b.ge	.LBB10_19
.LBB10_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB10_17 Depth 2
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark8heapsortIPddEEvT_S2_
	mov	x8, x23
.LBB10_17:                              //   Parent Loop BB10_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB10_15
// %bb.18:                              //   in Loop: Header=BB10_17 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB10_17
	b	.LBB10_14
.LBB10_19:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB10_20:
	ret
.Lfunc_end10:
	.size	_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc, .Lfunc_end10-_Z14test_heap_sortIPddEvT_S1_S1_S1_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,"axG",@progbits,_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,comdat
	.weak	_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc // -- Begin function _Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.p2align	2
	.type	_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc,@function
_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc: // @_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_startproc
// %bb.0:
	adrp	x8, iterations
	ldr	w8, [x8, :lo12:iterations]
	cmp	w8, #1
	b.lt	.LBB11_20
// %bb.1:
	sub	sp, sp, #128
	.cfi_def_cfa_offset 128
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
	mov	x19, x3
	mov	x20, x2
	mov	x21, x1
	mov	x22, x0
	cmp	x0, x1
	b.eq	.LBB11_13
// %bb.2:
	sub	x8, x21, x22
	mov	w10, #32                        // =0x20
	mov	w25, wzr
	sub	x8, x8, #8
	add	x26, x20, #8
	lsr	x9, x8, #3
	cmp	x8, #24
	add	x11, x9, #1
	sub	x9, x20, x22
	and	x27, x11, #0x3ffffffffffffffc
	stur	x11, [x29, #-8]                 // 8-byte Folded Spill
	add	x11, x20, #16
	lsl	x8, x27, #3
	ccmp	x9, x10, #0, hs
	str	x11, [sp, #16]                  // 8-byte Folded Spill
	add	x11, x22, #16
	cset	w28, lo
	add	x23, x22, x8
	add	x24, x20, x8
	str	x11, [sp, #8]                   // 8-byte Folded Spill
	b	.LBB11_5
.LBB11_3:                               //   in Loop: Header=BB11_5 Depth=1
	adrp	x8, current_test
	adrp	x0, .L.str.52
	add	x0, x0, :lo12:.L.str.52
	ldr	w1, [x8, :lo12:current_test]
	bl	printf
.LBB11_4:                               //   in Loop: Header=BB11_5 Depth=1
	adrp	x8, iterations
	add	w25, w25, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w25, w8
	b.ge	.LBB11_19
.LBB11_5:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_7 Depth 2
                                        //     Child Loop BB11_9 Depth 2
                                        //     Child Loop BB11_11 Depth 2
	mov	x8, x22
	mov	x9, x20
	tbnz	w28, #0, .LBB11_9
// %bb.6:                               // %.preheader6
                                        //   in Loop: Header=BB11_5 Depth=1
	ldp	x8, x9, [sp, #8]                // 16-byte Folded Reload
	mov	x10, x27
.LBB11_7:                               //   Parent Loop BB11_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q1, [x8, #-16]
	subs	x10, x10, #4
	add	x8, x8, #32
	stp	q0, q1, [x9, #-16]
	add	x9, x9, #32
	b.ne	.LBB11_7
// %bb.8:                               //   in Loop: Header=BB11_5 Depth=1
	ldur	x8, [x29, #-8]                  // 8-byte Folded Reload
	mov	x9, x24
	cmp	x8, x27
	mov	x8, x23
	b.eq	.LBB11_10
.LBB11_9:                               //   Parent Loop BB11_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d0, [x8], #8
	cmp	x8, x21
	str	d0, [x9], #8
	b.ne	.LBB11_9
.LBB11_10:                              //   in Loop: Header=BB11_5 Depth=1
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	mov	x8, x26
.LBB11_11:                              //   Parent Loop BB11_5 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB11_4
// %bb.12:                              //   in Loop: Header=BB11_11 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB11_11
	b	.LBB11_3
.LBB11_13:                              // %.preheader
	mov	w22, wzr
	add	x23, x20, #8
	adrp	x24, current_test
	adrp	x21, .L.str.52
	add	x21, x21, :lo12:.L.str.52
	b	.LBB11_16
.LBB11_14:                              //   in Loop: Header=BB11_16 Depth=1
	ldr	w1, [x24, :lo12:current_test]
	mov	x0, x21
	bl	printf
.LBB11_15:                              //   in Loop: Header=BB11_16 Depth=1
	adrp	x8, iterations
	add	w22, w22, #1
	ldr	w8, [x8, :lo12:iterations]
	cmp	w22, w8
	b.ge	.LBB11_19
.LBB11_16:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB11_17 Depth 2
	mov	x0, x20
	mov	x1, x19
	bl	_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	mov	x8, x23
.LBB11_17:                              //   Parent Loop BB11_16 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	cmp	x8, x19
	b.eq	.LBB11_15
// %bb.18:                              //   in Loop: Header=BB11_17 Depth=2
	ldp	d1, d0, [x8, #-8]
	add	x8, x8, #8
	fcmp	d0, d1
	b.pl	.LBB11_17
	b	.LBB11_14
.LBB11_19:
	.cfi_def_cfa wsp, 128
	ldp	x20, x19, [sp, #112]            // 16-byte Folded Reload
	ldp	x22, x21, [sp, #96]             // 16-byte Folded Reload
	ldp	x24, x23, [sp, #80]             // 16-byte Folded Reload
	ldp	x26, x25, [sp, #64]             // 16-byte Folded Reload
	ldp	x28, x27, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
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
.LBB11_20:
	ret
.Lfunc_end11:
	.size	_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc, .Lfunc_end11-_Z14test_heap_sortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEvT_S7_S7_S7_T0_PKc
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortIPddEEvT_S2_,"axG",@progbits,_ZN9benchmark9quicksortIPddEEvT_S2_,comdat
	.weak	_ZN9benchmark9quicksortIPddEEvT_S2_ // -- Begin function _ZN9benchmark9quicksortIPddEEvT_S2_
	.p2align	2
	.type	_ZN9benchmark9quicksortIPddEEvT_S2_,@function
_ZN9benchmark9quicksortIPddEEvT_S2_:    // @_ZN9benchmark9quicksortIPddEEvT_S2_
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
	bl	_ZN9benchmark9quicksortIPddEEvT_S2_
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
	.size	_ZN9benchmark9quicksortIPddEEvT_S2_, .Lfunc_end12-_ZN9benchmark9quicksortIPddEEvT_S2_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,"axG",@progbits,_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,comdat
	.weak	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_ // -- Begin function _ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	.p2align	2
	.type	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,@function
_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_: // @_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
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
	bl	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	cmp	x22, #8
	mov	x0, x20
	b.le	.LBB13_11
.LBB13_3:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB13_4 Depth 2
                                        //       Child Loop BB13_5 Depth 3
                                        //       Child Loop BB13_8 Depth 3
	ldr	d0, [x0]
	mov	x9, x0
	mov	x8, x19
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
	.size	_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_, .Lfunc_end13-_ZN9benchmark9quicksortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_,"axG",@progbits,_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_,comdat
	.weak	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_ // -- Begin function _ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
	.p2align	2
	.type	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_,@function
_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_: // @_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0]
	ldr	x9, [x1]
	sub	x10, x8, x9
	cmp	x10, #9
	b.lt	.LBB14_9
// %bb.1:
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             // 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	ldur	d0, [x8, #-8]
	mov	x19, x1
	mov	x10, x8
.LBB14_2:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB14_5 Depth 2
	ldr	d1, [x9], #8
	fcmp	d0, d1
	b.mi	.LBB14_2
// %bb.3:                               //   in Loop: Header=BB14_2 Depth=1
	cmp	x9, x10
	b.hs	.LBB14_8
// %bb.4:                               // %.preheader
                                        //   in Loop: Header=BB14_2 Depth=1
	add	x10, x10, #8
.LBB14_5:                               //   Parent Loop BB14_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldur	d2, [x10, #-16]
	sub	x10, x10, #8
	fcmp	d2, d0
	b.mi	.LBB14_5
// %bb.6:                               //   in Loop: Header=BB14_2 Depth=1
	cmp	x9, x10
	b.hs	.LBB14_8
// %bb.7:                               //   in Loop: Header=BB14_2 Depth=1
	stur	d2, [x9, #-8]
	stur	d1, [x10, #-8]
	b	.LBB14_2
.LBB14_8:
	sub	x20, x9, #8
	sub	x0, x29, #8
	add	x1, sp, #16
	stur	x8, [x29, #-8]
	str	x20, [sp, #16]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
	ldr	x8, [x19]
	add	x0, sp, #8
	mov	x1, sp
	stp	x8, x20, [sp]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #64
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB14_9:
	ret
.Lfunc_end14:
	.size	_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_, .Lfunc_end14-_ZN9benchmark9quicksortISt16reverse_iteratorIPdEdEEvT_S4_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,"axG",@progbits,_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,comdat
	.weak	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_ // -- Begin function _ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.p2align	2
	.type	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,@function
_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_: // @_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0]
	ldr	x9, [x1]
	sub	x10, x8, x9
	cmp	x10, #9
	b.lt	.LBB15_9
// %bb.1:
	sub	sp, sp, #64
	.cfi_def_cfa_offset 64
	stp	x29, x30, [sp, #32]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #48]             // 16-byte Folded Spill
	add	x29, sp, #32
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	ldur	d0, [x8, #-8]
	mov	x19, x1
	mov	x10, x8
.LBB15_2:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB15_5 Depth 2
	ldr	d1, [x9], #8
	fcmp	d0, d1
	b.mi	.LBB15_2
// %bb.3:                               //   in Loop: Header=BB15_2 Depth=1
	cmp	x9, x10
	b.hs	.LBB15_8
// %bb.4:                               // %.preheader
                                        //   in Loop: Header=BB15_2 Depth=1
	add	x10, x10, #8
.LBB15_5:                               //   Parent Loop BB15_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldur	d2, [x10, #-16]
	sub	x10, x10, #8
	fcmp	d2, d0
	b.mi	.LBB15_5
// %bb.6:                               //   in Loop: Header=BB15_2 Depth=1
	cmp	x9, x10
	b.hs	.LBB15_8
// %bb.7:                               //   in Loop: Header=BB15_2 Depth=1
	stur	d2, [x9, #-8]
	stur	d1, [x10, #-8]
	b	.LBB15_2
.LBB15_8:
	sub	x20, x9, #8
	sub	x0, x29, #8
	add	x1, sp, #16
	stur	x8, [x29, #-8]
	str	x20, [sp, #16]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	ldr	x8, [x19]
	add	x0, sp, #8
	mov	x1, sp
	stp	x8, x20, [sp]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.cfi_def_cfa wsp, 64
	ldp	x20, x19, [sp, #48]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #32]             // 16-byte Folded Reload
	add	sp, sp, #64
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB15_9:
	ret
.Lfunc_end15:
	.size	_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_, .Lfunc_end15-_ZN9benchmark9quicksortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,"axG",@progbits,_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,comdat
	.weak	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_ // -- Begin function _ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.p2align	2
	.type	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,@function
_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_: // @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0, #8]
	ldr	x9, [x1, #8]
	sub	x10, x9, x8
	cmp	x10, #9
	b.lt	.LBB16_9
// %bb.1:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #64]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             // 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	ldr	d0, [x8]
	mov	x19, x1
	mov	x10, x8
.LBB16_2:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB16_5 Depth 2
	ldr	d1, [x9, #-8]!
	fcmp	d0, d1
	b.mi	.LBB16_2
// %bb.3:                               //   in Loop: Header=BB16_2 Depth=1
	cmp	x10, x9
	b.hs	.LBB16_8
// %bb.4:                               // %.preheader
                                        //   in Loop: Header=BB16_2 Depth=1
	sub	x10, x10, #8
.LBB16_5:                               //   Parent Loop BB16_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x10, #8]!
	fcmp	d2, d0
	b.mi	.LBB16_5
// %bb.6:                               //   in Loop: Header=BB16_2 Depth=1
	cmp	x10, x9
	b.hs	.LBB16_8
// %bb.7:                               //   in Loop: Header=BB16_2 Depth=1
	str	d2, [x9]
	str	d1, [x10]
	b	.LBB16_2
.LBB16_8:
	add	x20, x9, #8
	sub	x0, x29, #16
	add	x1, sp, #32
	stur	x8, [x29, #-8]
	str	x20, [sp, #40]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	ldr	x8, [x19, #8]
	add	x0, sp, #16
	mov	x1, sp
	str	x20, [sp, #24]
	str	x8, [sp, #8]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.cfi_def_cfa wsp, 96
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #64]             // 16-byte Folded Reload
	add	sp, sp, #96
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB16_9:
	ret
.Lfunc_end16:
	.size	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_, .Lfunc_end16-_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,"axG",@progbits,_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,comdat
	.weak	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_ // -- Begin function _ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.p2align	2
	.type	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,@function
_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_: // @_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0, #8]
	ldr	x9, [x1, #8]
	sub	x10, x9, x8
	cmp	x10, #9
	b.lt	.LBB17_10
// %bb.1:
	sub	sp, sp, #96
	.cfi_def_cfa_offset 96
	stp	x29, x30, [sp, #64]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             // 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	ldr	d0, [x8]
	mov	x19, x1
	mov	x10, x8
.LBB17_2:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB17_3 Depth 2
                                        //     Child Loop BB17_6 Depth 2
	add	x20, x9, #8
.LBB17_3:                               //   Parent Loop BB17_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d1, [x9, #-8]!
	sub	x20, x20, #8
	fcmp	d0, d1
	b.mi	.LBB17_3
// %bb.4:                               //   in Loop: Header=BB17_2 Depth=1
	cmp	x10, x9
	b.hs	.LBB17_9
// %bb.5:                               // %.preheader
                                        //   in Loop: Header=BB17_2 Depth=1
	sub	x10, x10, #8
.LBB17_6:                               //   Parent Loop BB17_2 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	d2, [x10, #8]!
	fcmp	d2, d0
	b.mi	.LBB17_6
// %bb.7:                               //   in Loop: Header=BB17_2 Depth=1
	cmp	x10, x9
	b.hs	.LBB17_9
// %bb.8:                               //   in Loop: Header=BB17_2 Depth=1
	str	d2, [x9]
	str	d1, [x10]
	b	.LBB17_2
.LBB17_9:
	sub	x0, x29, #16
	add	x1, sp, #32
	stur	x8, [x29, #-8]
	str	x20, [sp, #40]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	ldr	x8, [x19, #8]
	add	x0, sp, #16
	mov	x1, sp
	str	x20, [sp, #24]
	str	x8, [sp, #8]
	bl	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.cfi_def_cfa wsp, 96
	ldp	x20, x19, [sp, #80]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #64]             // 16-byte Folded Reload
	add	sp, sp, #96
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
.LBB17_10:
	ret
.Lfunc_end17:
	.size	_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_, .Lfunc_end17-_ZN9benchmark9quicksortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortIPddEEvT_S2_,"axG",@progbits,_ZN9benchmark8heapsortIPddEEvT_S2_,comdat
	.weak	_ZN9benchmark8heapsortIPddEEvT_S2_ // -- Begin function _ZN9benchmark8heapsortIPddEEvT_S2_
	.p2align	2
	.type	_ZN9benchmark8heapsortIPddEEvT_S2_,@function
_ZN9benchmark8heapsortIPddEEvT_S2_:     // @_ZN9benchmark8heapsortIPddEEvT_S2_
	.cfi_startproc
// %bb.0:
	sub	x8, x1, x0
	asr	x8, x8, #3
	cmp	x8, #2
	b.lt	.LBB18_26
// %bb.1:
	lsr	x9, x8, #1
	sub	x10, x8, #1
	b	.LBB18_4
.LBB18_2:                               //   in Loop: Header=BB18_4 Depth=1
	mov	x13, x12
.LBB18_3:                               //   in Loop: Header=BB18_4 Depth=1
	cmp	x11, #1
	str	d0, [x0, x13, lsl #3]
	b.le	.LBB18_17
.LBB18_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_7 Depth 2
                                        //     Child Loop BB18_13 Depth 2
	mov	x11, x9
	sub	x9, x9, #1
	lsl	x12, x9, #1
	ldr	d0, [x0, x9, lsl #3]
	add	x13, x12, #2
	cmp	x13, x8
	b.ge	.LBB18_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB18_4 Depth=1
	mov	x14, x9
	b	.LBB18_7
.LBB18_6:                               // %select.end
                                        //   in Loop: Header=BB18_7 Depth=2
	sub	x12, x13, #1
	lsl	x13, x13, #1
	ldr	d1, [x0, x12, lsl #3]
	cmp	x13, x8
	str	d1, [x0, x14, lsl #3]
	mov	x14, x12
	b.ge	.LBB18_10
.LBB18_7:                               //   Parent Loop BB18_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x12, x0, x13, lsl #3
	ldp	d1, d2, [x12, #-8]
	fcmp	d1, d2
	b.pl	.LBB18_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB18_7 Depth=2
	add	x13, x13, #1
	b	.LBB18_6
.LBB18_9:                               //   in Loop: Header=BB18_4 Depth=1
	mov	x12, x9
.LBB18_10:                              //   in Loop: Header=BB18_4 Depth=1
	cmp	x13, x8
	b.ne	.LBB18_12
// %bb.11:                              //   in Loop: Header=BB18_4 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	str	d1, [x0, x12, lsl #3]
	mov	x12, x10
.LBB18_12:                              //   in Loop: Header=BB18_4 Depth=1
	cmp	x12, x11
	b.lt	.LBB18_2
.LBB18_13:                              //   Parent Loop BB18_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB18_2
// %bb.14:                              //   in Loop: Header=BB18_13 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.ge	.LBB18_13
	b	.LBB18_3
.LBB18_15:                              //   in Loop: Header=BB18_17 Depth=1
	mov	x10, xzr
.LBB18_16:                              //   in Loop: Header=BB18_17 Depth=1
	cmp	x8, #2
	mov	x8, x9
	str	d0, [x0, x10, lsl #3]
	b.le	.LBB18_26
.LBB18_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB18_19 Depth 2
                                        //     Child Loop BB18_24 Depth 2
	sub	x9, x8, #1
	ldr	d1, [x0]
	ldr	d0, [x0, x9, lsl #3]
	cmp	x9, #3
	str	d1, [x0, x9, lsl #3]
	b.lo	.LBB18_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB18_17 Depth=1
	mov	x12, xzr
	mov	w11, #2                         // =0x2
.LBB18_19:                              //   Parent Loop BB18_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x10, x0, x11, lsl #3
	ldp	d1, d2, [x10, #-8]
	fcmp	d1, d2
	cinc	x10, x11, mi
	lsl	x11, x10, #1
	sub	x10, x10, #1
	ldr	d1, [x0, x10, lsl #3]
	cmp	x11, x9
	str	d1, [x0, x12, lsl #3]
	mov	x12, x10
	b.lt	.LBB18_19
// %bb.20:                              //   in Loop: Header=BB18_17 Depth=1
	cmp	x11, x9
	b.ne	.LBB18_23
.LBB18_21:                              //   in Loop: Header=BB18_17 Depth=1
	sub	x11, x8, #2
	ldr	d1, [x0, x11, lsl #3]
	str	d1, [x0, x10, lsl #3]
	mov	x10, x11
	b	.LBB18_24
.LBB18_22:                              //   in Loop: Header=BB18_17 Depth=1
	mov	x10, xzr
	mov	w11, #2                         // =0x2
	cmp	x11, x9
	b.eq	.LBB18_21
.LBB18_23:                              //   in Loop: Header=BB18_17 Depth=1
	cmp	x10, #1
	b.lt	.LBB18_16
.LBB18_24:                              //   Parent Loop BB18_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB18_16
// %bb.25:                              //   in Loop: Header=BB18_24 Depth=2
	cmp	x12, #1
	str	d1, [x0, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB18_24
	b	.LBB18_15
.LBB18_26:
	ret
.Lfunc_end18:
	.size	_ZN9benchmark8heapsortIPddEEvT_S2_, .Lfunc_end18-_ZN9benchmark8heapsortIPddEEvT_S2_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,"axG",@progbits,_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,comdat
	.weak	_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_ // -- Begin function _ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	.p2align	2
	.type	_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_,@function
_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_: // @_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	.cfi_startproc
// %bb.0:
	sub	x8, x1, x0
	asr	x8, x8, #3
	cmp	x8, #2
	b.lt	.LBB19_26
// %bb.1:
	lsr	x9, x8, #1
	sub	x10, x8, #1
	b	.LBB19_4
.LBB19_2:                               //   in Loop: Header=BB19_4 Depth=1
	mov	x13, x12
.LBB19_3:                               //   in Loop: Header=BB19_4 Depth=1
	cmp	x11, #1
	str	d0, [x0, x13, lsl #3]
	b.le	.LBB19_17
.LBB19_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_7 Depth 2
                                        //     Child Loop BB19_13 Depth 2
	mov	x11, x9
	sub	x9, x9, #1
	lsl	x12, x9, #1
	ldr	d0, [x0, x9, lsl #3]
	add	x13, x12, #2
	cmp	x13, x8
	b.ge	.LBB19_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB19_4 Depth=1
	mov	x14, x9
	b	.LBB19_7
.LBB19_6:                               // %select.end
                                        //   in Loop: Header=BB19_7 Depth=2
	sub	x12, x13, #1
	lsl	x13, x13, #1
	ldr	d1, [x0, x12, lsl #3]
	cmp	x13, x8
	str	d1, [x0, x14, lsl #3]
	mov	x14, x12
	b.ge	.LBB19_10
.LBB19_7:                               //   Parent Loop BB19_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x12, x0, x13, lsl #3
	ldp	d1, d2, [x12, #-8]
	fcmp	d1, d2
	b.pl	.LBB19_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB19_7 Depth=2
	add	x13, x13, #1
	b	.LBB19_6
.LBB19_9:                               //   in Loop: Header=BB19_4 Depth=1
	mov	x12, x9
.LBB19_10:                              //   in Loop: Header=BB19_4 Depth=1
	cmp	x13, x8
	b.ne	.LBB19_12
// %bb.11:                              //   in Loop: Header=BB19_4 Depth=1
	ldr	d1, [x0, x10, lsl #3]
	str	d1, [x0, x12, lsl #3]
	mov	x12, x10
.LBB19_12:                              //   in Loop: Header=BB19_4 Depth=1
	cmp	x12, x11
	b.lt	.LBB19_2
.LBB19_13:                              //   Parent Loop BB19_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x12, #1
	add	x13, x13, x13, lsr #63
	asr	x13, x13, #1
	ldr	d1, [x0, x13, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_2
// %bb.14:                              //   in Loop: Header=BB19_13 Depth=2
	cmp	x13, x11
	str	d1, [x0, x12, lsl #3]
	mov	x12, x13
	b.ge	.LBB19_13
	b	.LBB19_3
.LBB19_15:                              //   in Loop: Header=BB19_17 Depth=1
	mov	x10, xzr
.LBB19_16:                              //   in Loop: Header=BB19_17 Depth=1
	cmp	x8, #2
	mov	x8, x9
	str	d0, [x0, x10, lsl #3]
	b.le	.LBB19_26
.LBB19_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB19_19 Depth 2
                                        //     Child Loop BB19_24 Depth 2
	sub	x9, x8, #1
	ldr	d1, [x0]
	ldr	d0, [x0, x9, lsl #3]
	cmp	x9, #3
	str	d1, [x0, x9, lsl #3]
	b.lo	.LBB19_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB19_17 Depth=1
	mov	x12, xzr
	mov	w11, #2                         // =0x2
.LBB19_19:                              //   Parent Loop BB19_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x10, x0, x11, lsl #3
	ldp	d1, d2, [x10, #-8]
	fcmp	d1, d2
	cinc	x10, x11, mi
	lsl	x11, x10, #1
	sub	x10, x10, #1
	ldr	d1, [x0, x10, lsl #3]
	cmp	x11, x9
	str	d1, [x0, x12, lsl #3]
	mov	x12, x10
	b.lt	.LBB19_19
// %bb.20:                              //   in Loop: Header=BB19_17 Depth=1
	cmp	x11, x9
	b.ne	.LBB19_23
.LBB19_21:                              //   in Loop: Header=BB19_17 Depth=1
	sub	x11, x8, #2
	ldr	d1, [x0, x11, lsl #3]
	str	d1, [x0, x10, lsl #3]
	mov	x10, x11
	b	.LBB19_24
.LBB19_22:                              //   in Loop: Header=BB19_17 Depth=1
	mov	x10, xzr
	mov	w11, #2                         // =0x2
	cmp	x11, x9
	b.eq	.LBB19_21
.LBB19_23:                              //   in Loop: Header=BB19_17 Depth=1
	cmp	x10, #1
	b.lt	.LBB19_16
.LBB19_24:                              //   Parent Loop BB19_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x12, x10, #1
	lsr	x11, x12, #1
	ldr	d1, [x0, x11, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB19_16
// %bb.25:                              //   in Loop: Header=BB19_24 Depth=2
	cmp	x12, #1
	str	d1, [x0, x10, lsl #3]
	mov	x10, x11
	b.hi	.LBB19_24
	b	.LBB19_15
.LBB19_26:
	ret
.Lfunc_end19:
	.size	_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_, .Lfunc_end19-_ZN9benchmark8heapsortIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEdEEvT_S8_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_,"axG",@progbits,_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_,comdat
	.weak	_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_ // -- Begin function _ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_
	.p2align	2
	.type	_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_,@function
_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_: // @_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0]
	ldr	x9, [x1]
	sub	x9, x8, x9
	asr	x9, x9, #3
	cmp	x9, #2
	b.lt	.LBB20_27
// %bb.1:
	mov	w10, #1                         // =0x1
	lsr	x12, x9, #1
	sub	x13, x9, #1
	sub	x11, x10, x9
	add	x11, x8, x11, lsl #3
	b	.LBB20_4
.LBB20_2:                               //   in Loop: Header=BB20_4 Depth=1
	mov	x15, x16
.LBB20_3:                               //   in Loop: Header=BB20_4 Depth=1
	sub	x15, x8, x15, lsl #3
	cmp	x14, #1
	stur	d0, [x15, #-8]
	b.le	.LBB20_15
.LBB20_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_7 Depth 2
                                        //     Child Loop BB20_13 Depth 2
	mov	x14, x12
	sub	x12, x12, #1
	sub	x15, x10, x14
	lsl	x16, x12, #1
	add	x17, x8, x15, lsl #3
	add	x15, x16, #2
	ldur	d0, [x17, #-8]
	cmp	x15, x9
	b.ge	.LBB20_9
// %bb.5:                               // %.preheader8
                                        //   in Loop: Header=BB20_4 Depth=1
	mov	x17, x12
	b	.LBB20_7
.LBB20_6:                               // %select.end
                                        //   in Loop: Header=BB20_7 Depth=2
	sub	x16, x10, x15
	sub	x17, x8, x17, lsl #3
	add	x18, x8, x16, lsl #3
	sub	x16, x15, #1
	lsl	x15, x15, #1
	ldur	d1, [x18, #-8]
	cmp	x15, x9
	stur	d1, [x17, #-8]
	mov	x17, x16
	b.ge	.LBB20_10
.LBB20_7:                               //   Parent Loop BB20_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x16, x10, x15
	sub	x18, x8, x15, lsl #3
	add	x16, x8, x16, lsl #3
	ldur	d2, [x18, #-8]
	ldur	d1, [x16, #-8]
	fcmp	d1, d2
	b.pl	.LBB20_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB20_7 Depth=2
	add	x15, x15, #1
	b	.LBB20_6
.LBB20_9:                               //   in Loop: Header=BB20_4 Depth=1
	mov	x16, x12
.LBB20_10:                              //   in Loop: Header=BB20_4 Depth=1
	cmp	x15, x9
	b.ne	.LBB20_12
// %bb.11:                              //   in Loop: Header=BB20_4 Depth=1
	ldur	d1, [x11, #-8]
	sub	x15, x8, x16, lsl #3
	mov	x16, x13
	stur	d1, [x15, #-8]
.LBB20_12:                              //   in Loop: Header=BB20_4 Depth=1
	cmp	x16, x14
	b.lt	.LBB20_2
.LBB20_13:                              //   Parent Loop BB20_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x15, x16, #1
	add	x15, x15, x15, lsr #63
	asr	x15, x15, #1
	sub	x17, x8, x15, lsl #3
	ldur	d1, [x17, #-8]
	fcmp	d1, d0
	b.pl	.LBB20_2
// %bb.14:                              //   in Loop: Header=BB20_13 Depth=2
	sub	x16, x8, x16, lsl #3
	cmp	x15, x14
	stur	d1, [x16, #-8]
	mov	x16, x15
	b.ge	.LBB20_13
	b	.LBB20_3
.LBB20_15:
	mov	w10, #1                         // =0x1
	mov	w11, #2                         // =0x2
	b	.LBB20_18
.LBB20_16:                              //   in Loop: Header=BB20_18 Depth=1
	mov	x13, xzr
.LBB20_17:                              //   in Loop: Header=BB20_18 Depth=1
	sub	x13, x8, x13, lsl #3
	cmp	x9, #2
	mov	x9, x12
	stur	d0, [x13, #-8]
	b.le	.LBB20_27
.LBB20_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB20_20 Depth 2
                                        //     Child Loop BB20_25 Depth 2
	sub	x12, x10, x9
	ldur	d1, [x8, #-8]
	add	x13, x8, x12, lsl #3
	sub	x12, x9, #1
	cmp	x12, #3
	ldur	d0, [x13, #-8]
	stur	d1, [x13, #-8]
	b.lo	.LBB20_23
// %bb.19:                              // %.preheader2
                                        //   in Loop: Header=BB20_18 Depth=1
	mov	x15, xzr
	mov	w14, #2                         // =0x2
.LBB20_20:                              //   Parent Loop BB20_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x10, x14
	sub	x16, x8, x14, lsl #3
	sub	x15, x8, x15, lsl #3
	add	x13, x8, x13, lsl #3
	ldur	d2, [x16, #-8]
	ldur	d1, [x13, #-8]
	fcmp	d1, d2
	cinc	x13, x14, mi
	sub	x14, x10, x13
	add	x16, x8, x14, lsl #3
	lsl	x14, x13, #1
	sub	x13, x13, #1
	ldur	d1, [x16, #-8]
	cmp	x14, x12
	stur	d1, [x15, #-8]
	mov	x15, x13
	b.lt	.LBB20_20
// %bb.21:                              //   in Loop: Header=BB20_18 Depth=1
	cmp	x14, x12
	b.ne	.LBB20_24
.LBB20_22:                              //   in Loop: Header=BB20_18 Depth=1
	sub	x14, x11, x9
	sub	x13, x8, x13, lsl #3
	add	x14, x8, x14, lsl #3
	ldur	d1, [x14, #-8]
	sub	x14, x9, #2
	stur	d1, [x13, #-8]
	mov	x13, x14
	b	.LBB20_25
.LBB20_23:                              //   in Loop: Header=BB20_18 Depth=1
	mov	x13, xzr
	mov	w14, #2                         // =0x2
	cmp	x14, x12
	b.eq	.LBB20_22
.LBB20_24:                              //   in Loop: Header=BB20_18 Depth=1
	cmp	x13, #1
	b.lt	.LBB20_17
.LBB20_25:                              //   Parent Loop BB20_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x15, x13, #1
	lsr	x14, x15, #1
	sub	x16, x8, x14, lsl #3
	ldur	d1, [x16, #-8]
	fcmp	d1, d0
	b.pl	.LBB20_17
// %bb.26:                              //   in Loop: Header=BB20_25 Depth=2
	sub	x13, x8, x13, lsl #3
	cmp	x15, #1
	stur	d1, [x13, #-8]
	mov	x13, x14
	b.hi	.LBB20_25
	b	.LBB20_16
.LBB20_27:
	ret
.Lfunc_end20:
	.size	_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_, .Lfunc_end20-_ZN9benchmark8heapsortISt16reverse_iteratorIPdEdEEvT_S4_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,"axG",@progbits,_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,comdat
	.weak	_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_ // -- Begin function _ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.p2align	2
	.type	_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_,@function
_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_: // @_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0]
	ldr	x9, [x1]
	sub	x9, x8, x9
	asr	x9, x9, #3
	cmp	x9, #2
	b.lt	.LBB21_27
// %bb.1:
	mov	w10, #1                         // =0x1
	lsr	x12, x9, #1
	sub	x13, x9, #1
	sub	x11, x10, x9
	add	x11, x8, x11, lsl #3
	b	.LBB21_4
.LBB21_2:                               //   in Loop: Header=BB21_4 Depth=1
	mov	x15, x16
.LBB21_3:                               //   in Loop: Header=BB21_4 Depth=1
	sub	x15, x8, x15, lsl #3
	cmp	x14, #1
	stur	d0, [x15, #-8]
	b.le	.LBB21_15
.LBB21_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_7 Depth 2
                                        //     Child Loop BB21_13 Depth 2
	mov	x14, x12
	sub	x12, x12, #1
	sub	x15, x10, x14
	lsl	x16, x12, #1
	add	x17, x8, x15, lsl #3
	add	x15, x16, #2
	ldur	d0, [x17, #-8]
	cmp	x15, x9
	b.ge	.LBB21_9
// %bb.5:                               // %.preheader8
                                        //   in Loop: Header=BB21_4 Depth=1
	mov	x17, x12
	b	.LBB21_7
.LBB21_6:                               // %select.end
                                        //   in Loop: Header=BB21_7 Depth=2
	sub	x16, x10, x15
	sub	x17, x8, x17, lsl #3
	add	x18, x8, x16, lsl #3
	sub	x16, x15, #1
	lsl	x15, x15, #1
	ldur	d1, [x18, #-8]
	cmp	x15, x9
	stur	d1, [x17, #-8]
	mov	x17, x16
	b.ge	.LBB21_10
.LBB21_7:                               //   Parent Loop BB21_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x16, x10, x15
	sub	x18, x8, x15, lsl #3
	add	x16, x8, x16, lsl #3
	ldur	d2, [x18, #-8]
	ldur	d1, [x16, #-8]
	fcmp	d1, d2
	b.pl	.LBB21_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB21_7 Depth=2
	add	x15, x15, #1
	b	.LBB21_6
.LBB21_9:                               //   in Loop: Header=BB21_4 Depth=1
	mov	x16, x12
.LBB21_10:                              //   in Loop: Header=BB21_4 Depth=1
	cmp	x15, x9
	b.ne	.LBB21_12
// %bb.11:                              //   in Loop: Header=BB21_4 Depth=1
	ldur	d1, [x11, #-8]
	sub	x15, x8, x16, lsl #3
	mov	x16, x13
	stur	d1, [x15, #-8]
.LBB21_12:                              //   in Loop: Header=BB21_4 Depth=1
	cmp	x16, x14
	b.lt	.LBB21_2
.LBB21_13:                              //   Parent Loop BB21_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x15, x16, #1
	add	x15, x15, x15, lsr #63
	asr	x15, x15, #1
	sub	x17, x8, x15, lsl #3
	ldur	d1, [x17, #-8]
	fcmp	d1, d0
	b.pl	.LBB21_2
// %bb.14:                              //   in Loop: Header=BB21_13 Depth=2
	sub	x16, x8, x16, lsl #3
	cmp	x15, x14
	stur	d1, [x16, #-8]
	mov	x16, x15
	b.ge	.LBB21_13
	b	.LBB21_3
.LBB21_15:
	mov	w10, #1                         // =0x1
	mov	w11, #2                         // =0x2
	b	.LBB21_18
.LBB21_16:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x13, xzr
.LBB21_17:                              //   in Loop: Header=BB21_18 Depth=1
	sub	x13, x8, x13, lsl #3
	cmp	x9, #2
	mov	x9, x12
	stur	d0, [x13, #-8]
	b.le	.LBB21_27
.LBB21_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB21_20 Depth 2
                                        //     Child Loop BB21_25 Depth 2
	sub	x12, x10, x9
	ldur	d1, [x8, #-8]
	add	x13, x8, x12, lsl #3
	sub	x12, x9, #1
	cmp	x12, #3
	ldur	d0, [x13, #-8]
	stur	d1, [x13, #-8]
	b.lo	.LBB21_23
// %bb.19:                              // %.preheader2
                                        //   in Loop: Header=BB21_18 Depth=1
	mov	x15, xzr
	mov	w14, #2                         // =0x2
.LBB21_20:                              //   Parent Loop BB21_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x10, x14
	sub	x16, x8, x14, lsl #3
	sub	x15, x8, x15, lsl #3
	add	x13, x8, x13, lsl #3
	ldur	d2, [x16, #-8]
	ldur	d1, [x13, #-8]
	fcmp	d1, d2
	cinc	x13, x14, mi
	sub	x14, x10, x13
	add	x16, x8, x14, lsl #3
	lsl	x14, x13, #1
	sub	x13, x13, #1
	ldur	d1, [x16, #-8]
	cmp	x14, x12
	stur	d1, [x15, #-8]
	mov	x15, x13
	b.lt	.LBB21_20
// %bb.21:                              //   in Loop: Header=BB21_18 Depth=1
	cmp	x14, x12
	b.ne	.LBB21_24
.LBB21_22:                              //   in Loop: Header=BB21_18 Depth=1
	sub	x14, x11, x9
	sub	x13, x8, x13, lsl #3
	add	x14, x8, x14, lsl #3
	ldur	d1, [x14, #-8]
	sub	x14, x9, #2
	stur	d1, [x13, #-8]
	mov	x13, x14
	b	.LBB21_25
.LBB21_23:                              //   in Loop: Header=BB21_18 Depth=1
	mov	x13, xzr
	mov	w14, #2                         // =0x2
	cmp	x14, x12
	b.eq	.LBB21_22
.LBB21_24:                              //   in Loop: Header=BB21_18 Depth=1
	cmp	x13, #1
	b.lt	.LBB21_17
.LBB21_25:                              //   Parent Loop BB21_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x15, x13, #1
	lsr	x14, x15, #1
	sub	x16, x8, x14, lsl #3
	ldur	d1, [x16, #-8]
	fcmp	d1, d0
	b.pl	.LBB21_17
// %bb.26:                              //   in Loop: Header=BB21_25 Depth=2
	sub	x13, x8, x13, lsl #3
	cmp	x15, #1
	stur	d1, [x13, #-8]
	mov	x13, x14
	b.hi	.LBB21_25
	b	.LBB21_16
.LBB21_27:
	ret
.Lfunc_end21:
	.size	_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_, .Lfunc_end21-_ZN9benchmark8heapsortISt16reverse_iteratorIN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEdEEvT_SA_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,"axG",@progbits,_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,comdat
	.weak	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_ // -- Begin function _ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.p2align	2
	.type	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_,@function
_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_: // @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.cfi_startproc
// %bb.0:
	ldr	x8, [x0, #8]
	ldr	x9, [x1, #8]
	sub	x9, x9, x8
	asr	x9, x9, #3
	cmp	x9, #2
	b.lt	.LBB22_26
// %bb.1:
	lsr	x10, x9, #1
	sub	x11, x9, #1
	b	.LBB22_4
.LBB22_2:                               //   in Loop: Header=BB22_4 Depth=1
	mov	x14, x13
.LBB22_3:                               //   in Loop: Header=BB22_4 Depth=1
	cmp	x12, #1
	str	d0, [x8, x14, lsl #3]
	b.le	.LBB22_17
.LBB22_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_7 Depth 2
                                        //     Child Loop BB22_13 Depth 2
	mov	x12, x10
	sub	x10, x10, #1
	lsl	x13, x10, #1
	ldr	d0, [x8, x10, lsl #3]
	add	x14, x13, #2
	cmp	x14, x9
	b.ge	.LBB22_9
// %bb.5:                               // %.preheader9
                                        //   in Loop: Header=BB22_4 Depth=1
	mov	x15, x10
	b	.LBB22_7
.LBB22_6:                               // %select.end
                                        //   in Loop: Header=BB22_7 Depth=2
	sub	x13, x14, #1
	lsl	x14, x14, #1
	ldr	d1, [x8, x13, lsl #3]
	cmp	x14, x9
	str	d1, [x8, x15, lsl #3]
	mov	x15, x13
	b.ge	.LBB22_10
.LBB22_7:                               //   Parent Loop BB22_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x13, x8, x14, lsl #3
	ldp	d1, d2, [x13, #-8]
	fcmp	d1, d2
	b.pl	.LBB22_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB22_7 Depth=2
	add	x14, x14, #1
	b	.LBB22_6
.LBB22_9:                               //   in Loop: Header=BB22_4 Depth=1
	mov	x13, x10
.LBB22_10:                              //   in Loop: Header=BB22_4 Depth=1
	cmp	x14, x9
	b.ne	.LBB22_12
// %bb.11:                              //   in Loop: Header=BB22_4 Depth=1
	ldr	d1, [x8, x11, lsl #3]
	str	d1, [x8, x13, lsl #3]
	mov	x13, x11
.LBB22_12:                              //   in Loop: Header=BB22_4 Depth=1
	cmp	x13, x12
	b.lt	.LBB22_2
.LBB22_13:                              //   Parent Loop BB22_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x14, x13, #1
	add	x14, x14, x14, lsr #63
	asr	x14, x14, #1
	ldr	d1, [x8, x14, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_2
// %bb.14:                              //   in Loop: Header=BB22_13 Depth=2
	cmp	x14, x12
	str	d1, [x8, x13, lsl #3]
	mov	x13, x14
	b.ge	.LBB22_13
	b	.LBB22_3
.LBB22_15:                              //   in Loop: Header=BB22_17 Depth=1
	mov	x11, xzr
.LBB22_16:                              //   in Loop: Header=BB22_17 Depth=1
	cmp	x9, #2
	mov	x9, x10
	str	d0, [x8, x11, lsl #3]
	b.le	.LBB22_26
.LBB22_17:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB22_19 Depth 2
                                        //     Child Loop BB22_24 Depth 2
	sub	x10, x9, #1
	ldr	d1, [x8]
	ldr	d0, [x8, x10, lsl #3]
	cmp	x10, #3
	str	d1, [x8, x10, lsl #3]
	b.lo	.LBB22_22
// %bb.18:                              // %.preheader2
                                        //   in Loop: Header=BB22_17 Depth=1
	mov	x13, xzr
	mov	w12, #2                         // =0x2
.LBB22_19:                              //   Parent Loop BB22_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x11, x8, x12, lsl #3
	ldp	d1, d2, [x11, #-8]
	fcmp	d1, d2
	cinc	x11, x12, mi
	lsl	x12, x11, #1
	sub	x11, x11, #1
	ldr	d1, [x8, x11, lsl #3]
	cmp	x12, x10
	str	d1, [x8, x13, lsl #3]
	mov	x13, x11
	b.lt	.LBB22_19
// %bb.20:                              //   in Loop: Header=BB22_17 Depth=1
	cmp	x12, x10
	b.ne	.LBB22_23
.LBB22_21:                              //   in Loop: Header=BB22_17 Depth=1
	sub	x12, x9, #2
	ldr	d1, [x8, x12, lsl #3]
	str	d1, [x8, x11, lsl #3]
	mov	x11, x12
	b	.LBB22_24
.LBB22_22:                              //   in Loop: Header=BB22_17 Depth=1
	mov	x11, xzr
	mov	w12, #2                         // =0x2
	cmp	x12, x10
	b.eq	.LBB22_21
.LBB22_23:                              //   in Loop: Header=BB22_17 Depth=1
	cmp	x11, #1
	b.lt	.LBB22_16
.LBB22_24:                              //   Parent Loop BB22_17 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x11, #1
	lsr	x12, x13, #1
	ldr	d1, [x8, x12, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB22_16
// %bb.25:                              //   in Loop: Header=BB22_24 Depth=2
	cmp	x13, #1
	str	d1, [x8, x11, lsl #3]
	mov	x11, x12
	b.hi	.LBB22_24
	b	.LBB22_15
.LBB22_26:
	ret
.Lfunc_end22:
	.size	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_, .Lfunc_end22-_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IPdEEdEEvT_S5_
	.cfi_endproc
                                        // -- End function
	.section	.text._ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,"axG",@progbits,_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,comdat
	.weak	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_ // -- Begin function _ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.p2align	2
	.type	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_,@function
_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_: // @_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
	.cfi_startproc
// %bb.0:
	ldr	x9, [x0, #8]
	ldr	x8, [x1, #8]
	sub	x8, x8, x9
	asr	x8, x8, #3
	cmp	x8, #2
	b.lt	.LBB23_27
// %bb.1:
	lsr	x10, x8, #1
	sub	x11, x8, #1
	b	.LBB23_4
.LBB23_2:                               //   in Loop: Header=BB23_4 Depth=1
	mov	x14, x13
.LBB23_3:                               //   in Loop: Header=BB23_4 Depth=1
	cmp	x12, #1
	str	d0, [x9, x14, lsl #3]
	b.le	.LBB23_15
.LBB23_4:                               // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_7 Depth 2
                                        //     Child Loop BB23_13 Depth 2
	mov	x12, x10
	sub	x10, x10, #1
	lsl	x13, x10, #1
	ldr	d0, [x9, x10, lsl #3]
	add	x14, x13, #2
	cmp	x14, x8
	b.ge	.LBB23_9
// %bb.5:                               // %.preheader8
                                        //   in Loop: Header=BB23_4 Depth=1
	mov	x15, x10
	b	.LBB23_7
.LBB23_6:                               // %select.end
                                        //   in Loop: Header=BB23_7 Depth=2
	sub	x13, x14, #1
	lsl	x14, x14, #1
	ldr	d1, [x9, x13, lsl #3]
	cmp	x14, x8
	str	d1, [x9, x15, lsl #3]
	mov	x15, x13
	b.ge	.LBB23_10
.LBB23_7:                               //   Parent Loop BB23_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x13, x9, x14, lsl #3
	ldp	d1, d2, [x13, #-8]
	fcmp	d1, d2
	b.pl	.LBB23_6
// %bb.8:                               // %select.true.sink
                                        //   in Loop: Header=BB23_7 Depth=2
	add	x14, x14, #1
	b	.LBB23_6
.LBB23_9:                               //   in Loop: Header=BB23_4 Depth=1
	mov	x13, x10
.LBB23_10:                              //   in Loop: Header=BB23_4 Depth=1
	cmp	x14, x8
	b.ne	.LBB23_12
// %bb.11:                              //   in Loop: Header=BB23_4 Depth=1
	ldr	d1, [x9, x11, lsl #3]
	str	d1, [x9, x13, lsl #3]
	mov	x13, x11
.LBB23_12:                              //   in Loop: Header=BB23_4 Depth=1
	cmp	x13, x12
	b.lt	.LBB23_2
.LBB23_13:                              //   Parent Loop BB23_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x14, x13, #1
	add	x14, x14, x14, lsr #63
	asr	x14, x14, #1
	ldr	d1, [x9, x14, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB23_2
// %bb.14:                              //   in Loop: Header=BB23_13 Depth=2
	cmp	x14, x12
	str	d1, [x9, x13, lsl #3]
	mov	x13, x14
	b.ge	.LBB23_13
	b	.LBB23_3
.LBB23_15:
	ldr	x9, [x0, #8]
	b	.LBB23_18
.LBB23_16:                              //   in Loop: Header=BB23_18 Depth=1
	mov	x11, xzr
.LBB23_17:                              //   in Loop: Header=BB23_18 Depth=1
	cmp	x8, #2
	mov	x8, x10
	str	d0, [x9, x11, lsl #3]
	b.le	.LBB23_27
.LBB23_18:                              // =>This Loop Header: Depth=1
                                        //     Child Loop BB23_20 Depth 2
                                        //     Child Loop BB23_25 Depth 2
	sub	x10, x8, #1
	ldr	d1, [x9]
	ldr	d0, [x9, x10, lsl #3]
	cmp	x10, #3
	str	d1, [x9, x10, lsl #3]
	b.lo	.LBB23_23
// %bb.19:                              // %.preheader2
                                        //   in Loop: Header=BB23_18 Depth=1
	mov	x13, xzr
	mov	w12, #2                         // =0x2
.LBB23_20:                              //   Parent Loop BB23_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	add	x11, x9, x12, lsl #3
	ldp	d1, d2, [x11, #-8]
	fcmp	d1, d2
	cinc	x11, x12, mi
	lsl	x12, x11, #1
	sub	x11, x11, #1
	ldr	d1, [x9, x11, lsl #3]
	cmp	x12, x10
	str	d1, [x9, x13, lsl #3]
	mov	x13, x11
	b.lt	.LBB23_20
// %bb.21:                              //   in Loop: Header=BB23_18 Depth=1
	cmp	x12, x10
	b.ne	.LBB23_24
.LBB23_22:                              //   in Loop: Header=BB23_18 Depth=1
	sub	x12, x8, #2
	ldr	d1, [x9, x12, lsl #3]
	str	d1, [x9, x11, lsl #3]
	mov	x11, x12
	b	.LBB23_25
.LBB23_23:                              //   in Loop: Header=BB23_18 Depth=1
	mov	x11, xzr
	mov	w12, #2                         // =0x2
	cmp	x12, x10
	b.eq	.LBB23_22
.LBB23_24:                              //   in Loop: Header=BB23_18 Depth=1
	cmp	x11, #1
	b.lt	.LBB23_17
.LBB23_25:                              //   Parent Loop BB23_18 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	sub	x13, x11, #1
	lsr	x12, x13, #1
	ldr	d1, [x9, x12, lsl #3]
	fcmp	d1, d0
	b.pl	.LBB23_17
// %bb.26:                              //   in Loop: Header=BB23_25 Depth=2
	cmp	x13, #1
	str	d1, [x9, x11, lsl #3]
	mov	x11, x12
	b.hi	.LBB23_25
	b	.LBB23_16
.LBB23_27:
	ret
.Lfunc_end23:
	.size	_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_, .Lfunc_end23-_ZN9benchmark8heapsortISt16reverse_iteratorIS1_IN9__gnu_cxx17__normal_iteratorIPdSt6vectorIdSaIdEEEEEEdEEvT_SB_
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
	.word	60000                           // 0xea60
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

	.type	dataMaster,@object              // @dataMaster
	.globl	dataMaster
	.p2align	3, 0x0
dataMaster:
	.zero	16000
	.size	dataMaster, 16000

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

	.type	rdpb,@object                    // @rdpb
	.globl	rdpb
	.p2align	3, 0x0
rdpb:
	.xword	data+16000
	.size	rdpb, 8

	.type	rdpe,@object                    // @rdpe
	.globl	rdpe
	.p2align	3, 0x0
rdpe:
	.xword	data
	.size	rdpe, 8

	.type	rdMpb,@object                   // @rdMpb
	.globl	rdMpb
	.p2align	3, 0x0
rdMpb:
	.xword	dataMaster+16000
	.size	rdMpb, 8

	.type	rdMpe,@object                   // @rdMpe
	.globl	rdMpe
	.p2align	3, 0x0
rdMpe:
	.xword	dataMaster
	.size	rdMpe, 8

	.type	rrdpb,@object                   // @rrdpb
	.globl	rrdpb
	.p2align	3, 0x0
rrdpb:
	.zero	8
	.xword	data
	.size	rrdpb, 16

	.type	rrdpe,@object                   // @rrdpe
	.globl	rrdpe
	.p2align	3, 0x0
rrdpe:
	.zero	8
	.xword	data+16000
	.size	rrdpe, 16

	.type	rrdMpb,@object                  // @rrdMpb
	.globl	rrdMpb
	.p2align	3, 0x0
rrdMpb:
	.zero	8
	.xword	dataMaster
	.size	rrdMpb, 16

	.type	rrdMpe,@object                  // @rrdMpe
	.globl	rrdMpe
	.p2align	3, 0x0
rrdMpe:
	.zero	8
	.xword	dataMaster+16000
	.size	rrdMpe, 16

	.type	.L.str.26,@object               // @.str.26
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.26:
	.asciz	"insertion_sort double pointer verify2"
	.size	.L.str.26, 38

	.type	.L.str.27,@object               // @.str.27
.L.str.27:
	.asciz	"insertion_sort double vector iterator"
	.size	.L.str.27, 38

	.type	.L.str.34,@object               // @.str.34
.L.str.34:
	.asciz	"quicksort double pointer verify2"
	.size	.L.str.34, 33

	.type	.L.str.35,@object               // @.str.35
.L.str.35:
	.asciz	"quicksort double vector iterator"
	.size	.L.str.35, 33

	.type	.L.str.42,@object               // @.str.42
.L.str.42:
	.asciz	"heap_sort double pointer verify2"
	.size	.L.str.42, 33

	.type	.L.str.43,@object               // @.str.43
.L.str.43:
	.asciz	"heap_sort double vector iterator"
	.size	.L.str.43, 33

	.type	.L.str.51,@object               // @.str.51
.L.str.51:
	.asciz	"test %i failed\n"
	.size	.L.str.51, 16

	.type	.L.str.52,@object               // @.str.52
.L.str.52:
	.asciz	"sort test %i failed\n"
	.size	.L.str.52, 21

	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3, 0x0
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.xword	__gxx_personality_v0
	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
