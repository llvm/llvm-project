	.file	"almabench.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function anpm
.LCPI0_0:
	.xword	0x401921fb54442d18              // double 6.2831853071795862
.LCPI0_2:
	.xword	0x400921fb54442d18              // double 3.1415926535897931
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	3, 0x0
.LCPI0_1:
	.xword	0x401921fb54442d18              // double 6.2831853071795862
	.xword	0xc01921fb54442d18              // double -6.2831853071795862
	.text
	.globl	anpm
	.p2align	2
	.type	anpm,@function
anpm:                                   // @anpm
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
	adrp	x8, .LCPI0_0
	fmov	d8, d0
	ldr	d1, [x8, :lo12:.LCPI0_0]
	bl	fmod
	fcmp	d8, #0.0
	mov	w8, #8                          // =0x8
	fabs	d1, d0
	adrp	x9, .LCPI0_1
	add	x9, x9, :lo12:.LCPI0_1
	csel	x8, x8, xzr, mi
	ldr	d2, [x9, x8]
	adrp	x8, .LCPI0_2
	ldr	d3, [x8, :lo12:.LCPI0_2]
	fsub	d2, d0, d2
	fcmp	d1, d3
	fcsel	d0, d0, d2, lt
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #32                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.Lfunc_end0:
	.size	anpm, .Lfunc_end0-anpm
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function planetpv
.LCPI1_0:
	.xword	0xc142b42c80000000              // double -2451545
.LCPI1_1:
	.xword	0x3ed455a5b2ff8f9d              // double 4.8481368110953598E-6
.LCPI1_2:
	.xword	0x401921fb54442d18              // double 6.2831853071795862
.LCPI1_3:
	.xword	0x3fd702a41f2e9970              // double 0.35953619999999997
.LCPI1_4:
	.xword	0x3e7ad7f29abcaf48              // double 9.9999999999999995E-8
.LCPI1_6:
	.xword	0x400921fb54442d18              // double 3.1415926535897931
.LCPI1_7:
	.xword	0x3d719799812dea11              // double 9.9999999999999998E-13
.LCPI1_8:
	.xword	0x3f919d6d51a6b69a              // double 0.017202098950000001
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0
.LCPI1_5:
	.xword	0x401921fb54442d18              // double 6.2831853071795862
	.xword	0xc01921fb54442d18              // double -6.2831853071795862
.LCPI1_9:
	.xword	0xbfd9752e50f4b399              // double -0.39777715593191371
	.xword	0x3fed5c0357681ef3              // double 0.91748206206918181
.LCPI1_10:
	.xword	0x3fed5c0357681ef3              // double 0.91748206206918181
	.xword	0x3fd9752e50f4b399              // double 0.39777715593191371
	.text
	.globl	planetpv
	.p2align	2
	.type	planetpv,@function
planetpv:                               // @planetpv
	.cfi_startproc
// %bb.0:
	stp	d15, d14, [sp, #-160]!          // 16-byte Folded Spill
	.cfi_def_cfa_offset 160
	stp	d13, d12, [sp, #16]             // 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             // 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               // 16-byte Folded Spill
	stp	x29, x30, [sp, #64]             // 16-byte Folded Spill
	str	x28, [sp, #80]                  // 8-byte Folded Spill
	stp	x26, x25, [sp, #96]             // 16-byte Folded Spill
	stp	x24, x23, [sp, #112]            // 16-byte Folded Spill
	stp	x22, x21, [sp, #128]            // 16-byte Folded Spill
	stp	x20, x19, [sp, #144]            // 16-byte Folded Spill
	add	x29, sp, #64
	.cfi_def_cfa w29, 96
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w23, -40
	.cfi_offset w24, -48
	.cfi_offset w25, -56
	.cfi_offset w26, -64
	.cfi_offset w28, -80
	.cfi_offset w30, -88
	.cfi_offset w29, -96
	.cfi_offset b8, -104
	.cfi_offset b9, -112
	.cfi_offset b10, -120
	.cfi_offset b11, -128
	.cfi_offset b12, -136
	.cfi_offset b13, -144
	.cfi_offset b14, -152
	.cfi_offset b15, -160
	.cfi_remember_state
	sub	sp, sp, #400
	adrp	x8, .LCPI1_0
	ldp	d1, d2, [x0]
	ldr	d0, [x8, :lo12:.LCPI1_0]
	mov	x8, #82497731821568             // =0x4b0800000000
	mov	w20, w1
	movk	x8, #16662, lsl #48
	sxtw	x21, w20
	adrp	x9, pi
	add	x9, x9, :lo12:pi
	fadd	d0, d1, d0
	fmov	d1, x8
	add	x8, x21, w1, sxtw #1
	mov	x19, x2
	lsl	x22, x8, #3
	adrp	x8, dlm
	add	x8, x8, :lo12:dlm
	fadd	d0, d0, d2
	add	x8, x8, x22
	add	x9, x9, x22
	ldp	d2, d3, [x9, #8]
	fdiv	d8, d0, d1
	ldp	d0, d1, [x8, #8]
	fmadd	d0, d1, d8, d0
	fmadd	d1, d3, d8, d2
	ldr	d2, [x8]
	mov	x8, #35184372088832             // =0x200000000000
	ldr	d3, [x9]
	movk	x8, #16556, lsl #48
	fmov	d9, x8
	adrp	x8, .LCPI1_1
	fmul	d0, d8, d0
	fmul	d1, d8, d1
	ldr	d10, [x8, :lo12:.LCPI1_1]
	adrp	x8, a
	add	x8, x8, :lo12:a
	add	x8, x8, x22
	fmadd	d0, d2, d9, d0
	fmadd	d1, d3, d9, d1
	ldr	d2, [x8, #16]
	ldp	d3, d4, [x8]
	adrp	x8, e
	add	x8, x8, :lo12:e
	add	x8, x8, x22
	stp	d2, d4, [x29, #-128]            // 16-byte Folded Spill
	fmul	d11, d0, d10
	fmul	d2, d1, d10
	ldp	d1, d0, [x8]
	stur	d3, [x29, #-88]                 // 8-byte Folded Spill
	str	d0, [sp, #136]                  // 8-byte Folded Spill
	ldr	d0, [x8, #16]
	adrp	x8, .LCPI1_2
	ldr	d12, [x8, :lo12:.LCPI1_2]
	str	d1, [x29, #24]                  // 8-byte Folded Spill
	str	d0, [sp, #96]                   // 8-byte Folded Spill
	fmov	d0, d2
	fmov	d1, d12
	stp	d12, d2, [sp, #72]              // 16-byte Folded Spill
	bl	fmod
	adrp	x8, omega
	add	x8, x8, :lo12:omega
	str	d0, [sp, #40]                   // 8-byte Folded Spill
	add	x8, x8, x22
	ldp	d1, d2, [x8, #8]
	fmadd	d1, d2, d8, d1
	ldr	d2, [x8]
	adrp	x8, dinc
	add	x8, x8, :lo12:dinc
	add	x8, x8, x22
	ldp	d3, d4, [x8]
	fmul	d1, d8, d1
	stp	d3, d10, [x29, #-80]            // 16-byte Folded Spill
	fmadd	d1, d2, d9, d1
	fmul	d2, d1, d10
	ldr	d1, [x8, #16]
	stp	d1, d4, [x29, #-112]            // 16-byte Folded Spill
	fmov	d1, d12
	fmov	d0, d2
	stur	d2, [x29, #-144]                // 8-byte Folded Spill
	bl	fmod
	adrp	x8, .LCPI1_3
	add	x9, x21, w20, sxtw #3
	stur	d0, [x29, #-96]                 // 8-byte Folded Spill
	ldr	d0, [x8, :lo12:.LCPI1_3]
	adrp	x10, kp
	add	x10, x10, :lo12:kp
	lsl	x8, x9, #3
	add	x11, x21, w20, sxtw #2
	adrp	x9, kq
	add	x9, x9, :lo12:kq
	fmul	d15, d8, d0
	add	x24, x10, x8
	lsl	x11, x11, #4
	adrp	x10, sa
	add	x10, x10, :lo12:sa
	ldr	d0, [x24]
	add	x20, x9, x11
	adrp	x9, ca
	add	x9, x9, :lo12:ca
	fmul	d9, d15, d0
	add	x26, x9, x8
	ldr	d1, [x20]
	ldr	d0, [x26]
	adrp	x9, cl
	add	x9, x9, :lo12:cl
	add	x25, x10, x8
	adrp	x8, sl
	add	x8, x8, :lo12:sl
	stur	d0, [x29, #-136]                // 8-byte Folded Spill
	fmul	d10, d15, d1
	add	x23, x9, x11
	fmov	d0, d9
	add	x22, x8, x11
	bl	cos
	stur	d0, [x29, #-152]                // 8-byte Folded Spill
	fmov	d0, d9
	ldr	d1, [x25]
	stur	d1, [x29, #-192]                // 8-byte Folded Spill
	bl	sin
	stur	d0, [x29, #-200]                // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d13, [x23]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d12, [x22]
	bl	sin
	fmul	d0, d12, d0
	ldr	d1, [x24, #8]
	adrp	x8, .LCPI1_4
	ldr	d2, [x20, #8]
	fmul	d10, d15, d2
	fmadd	d0, d13, d9, d0
	ldr	d13, [x8, :lo12:.LCPI1_4]
	fmul	d9, d15, d1
	fmadd	d11, d0, d13, d11
	ldr	d0, [x26, #8]
	stur	d0, [x29, #-160]                // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	ldr	d1, [x25, #8]
	stp	d1, d0, [x29, #-176]            // 16-byte Folded Spill
	fmov	d0, d9
	bl	sin
	stur	d0, [x29, #-184]                // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d12, [x23, #8]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d14, [x22, #8]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #16]
	ldr	d2, [x20, #16]
	fmul	d10, d15, d2
	fmadd	d0, d12, d9, d0
	fmul	d9, d15, d1
	fmadd	d11, d0, d13, d11
	ldr	d0, [x26, #16]
	stur	d0, [x29, #-208]                // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	ldr	d1, [x25, #16]
	stp	d1, d0, [x29, #-224]            // 16-byte Folded Spill
	fmov	d0, d9
	bl	sin
	str	d0, [sp, #232]                  // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d12, [x23, #16]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d14, [x22, #16]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #24]
	ldr	d2, [x20, #24]
	fmul	d10, d15, d2
	fmadd	d0, d12, d9, d0
	fmul	d9, d15, d1
	fmadd	d11, d0, d13, d11
	ldr	d0, [x26, #24]
	str	d0, [sp, #224]                  // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	ldr	d1, [x25, #24]
	stp	d1, d0, [sp, #208]              // 16-byte Folded Spill
	fmov	d0, d9
	bl	sin
	str	d0, [sp, #200]                  // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d12, [x23, #24]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d14, [x22, #24]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #32]
	ldr	d2, [x20, #32]
	fmul	d10, d15, d2
	fmadd	d0, d12, d9, d0
	fmul	d9, d15, d1
	fmadd	d11, d0, d13, d11
	ldr	d0, [x26, #32]
	str	d0, [sp, #192]                  // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	ldr	d1, [x25, #32]
	stp	d1, d0, [sp, #176]              // 16-byte Folded Spill
	fmov	d0, d9
	bl	sin
	str	d0, [sp, #168]                  // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d12, [x23, #32]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d14, [x22, #32]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #40]
	ldr	d2, [x20, #40]
	fmul	d10, d15, d2
	fmadd	d0, d12, d9, d0
	fmul	d9, d15, d1
	fmadd	d11, d0, d13, d11
	ldr	d0, [x26, #40]
	str	d0, [sp, #160]                  // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	ldr	d1, [x25, #40]
	stp	d1, d0, [sp, #144]              // 16-byte Folded Spill
	fmov	d0, d9
	bl	sin
	str	d0, [sp, #128]                  // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d12, [x23, #40]
	bl	cos
	fmov	d9, d0
	fmov	d0, d10
	ldr	d14, [x22, #40]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #48]
	ldr	d2, [x20, #48]
	fmul	d10, d15, d2
	fmadd	d0, d12, d9, d0
	fmul	d9, d15, d1
	fmadd	d12, d0, d13, d11
	ldr	d0, [x26, #48]
	str	d0, [sp, #120]                  // 8-byte Folded Spill
	fmov	d0, d9
	bl	cos
	str	d0, [sp, #112]                  // 8-byte Folded Spill
	fmov	d0, d9
	ldr	d1, [x25, #48]
	str	d1, [sp, #32]                   // 8-byte Folded Spill
	bl	sin
	str	d0, [sp, #16]                   // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d9, [x23, #48]
	bl	cos
	fmov	d11, d0
	fmov	d0, d10
	ldr	d14, [x22, #48]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #56]
	ldr	d2, [x20, #56]
	fmul	d10, d15, d1
	fmadd	d0, d9, d11, d0
	fmul	d11, d15, d2
	fmadd	d9, d0, d13, d12
	ldr	d0, [x26, #56]
	str	d0, [sp, #104]                  // 8-byte Folded Spill
	fmov	d0, d10
	bl	cos
	str	d0, [sp, #88]                   // 8-byte Folded Spill
	fmov	d0, d10
	ldr	d1, [x25, #56]
	str	d1, [sp, #8]                    // 8-byte Folded Spill
	bl	sin
	str	d0, [sp]                        // 8-byte Folded Spill
	fmov	d0, d11
	ldr	d12, [x23, #56]
	bl	cos
	fmov	d10, d0
	fmov	d0, d11
	ldr	d14, [x22, #56]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x24, #64]
	fmul	d11, d15, d1
	fmadd	d0, d12, d10, d0
	fmadd	d9, d0, d13, d9
	ldr	d0, [x26, #64]
	str	d0, [sp, #64]                   // 8-byte Folded Spill
	fmov	d0, d11
	bl	cos
	ldr	d1, [x25, #64]
	stp	d1, d0, [sp, #48]               // 16-byte Folded Spill
	fmov	d0, d11
	bl	sin
	ldr	d1, [x20, #64]
	str	d0, [sp, #24]                   // 8-byte Folded Spill
	ldr	d12, [x23, #64]
	fmul	d10, d15, d1
	fmov	d0, d10
	bl	cos
	fmov	d11, d0
	fmov	d0, d10
	ldr	d14, [x22, #64]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [x20, #72]
	fmul	d10, d15, d1
	fmadd	d0, d12, d11, d0
	ldr	d12, [x23, #72]
	fmul	d0, d8, d0
	fmadd	d9, d0, d13, d9
	fmov	d0, d10
	bl	cos
	fmov	d11, d0
	fmov	d0, d10
	ldr	d14, [x22, #72]
	bl	sin
	fmul	d0, d14, d0
	ldr	d1, [sp, #80]                   // 8-byte Folded Reload
	ldr	d5, [sp, #40]                   // 8-byte Folded Reload
	mov	w8, #8                          // =0x8
	adrp	x20, .LCPI1_5
	add	x20, x20, :lo12:.LCPI1_5
	fcmp	d1, #0.0
	fabs	d1, d5
	ldr	d3, [sp, #136]                  // 8-byte Folded Reload
	ldr	d4, [sp, #96]                   // 8-byte Folded Reload
	fmadd	d0, d12, d11, d0
	ldr	d11, [x29, #24]                 // 8-byte Folded Reload
	csel	x8, x8, xzr, mi
	fmadd	d10, d4, d8, d3
	ldr	d2, [x20, x8]
	adrp	x8, .LCPI1_6
	ldr	d3, [x8, :lo12:.LCPI1_6]
	fmul	d0, d8, d0
	fsub	d2, d5, d2
	fcmp	d1, d3
	ldr	d1, [sp, #72]                   // 8-byte Folded Reload
	str	d3, [sp, #96]                   // 8-byte Folded Spill
	fmadd	d15, d10, d8, d11
	fmadd	d0, d0, d13, d9
	fcsel	d9, d5, d2, lt
	bl	fmod
	fsub	d14, d0, d9
	str	d9, [sp, #136]                  // 8-byte Folded Spill
	fmov	d0, d14
	bl	sin
	fnmadd	d11, d10, d8, d11
	fmadd	d10, d15, d0, d14
	fmov	d0, d10
	fsub	d9, d14, d10
	bl	sin
	fmadd	d9, d15, d0, d9
	fmov	d0, d10
	bl	cos
	fmov	d5, #1.00000000
	adrp	x8, .LCPI1_7
	str	d11, [x29, #24]                 // 8-byte Folded Spill
	fmsub	d0, d15, d0, d5
	fdiv	d0, d9, d0
	ldr	d9, [x8, :lo12:.LCPI1_7]
	fabs	d1, d0
	fadd	d12, d10, d0
	fcmp	d1, d9
	b.mi	.LBB1_11
// %bb.1:
	fmov	d0, d12
	fsub	d10, d14, d12
	bl	sin
	fmadd	d10, d15, d0, d10
	fmov	d0, d12
	bl	cos
	fmov	d5, #1.00000000
	fmadd	d0, d11, d0, d5
	fdiv	d0, d10, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_11
// %bb.2:
	fmov	d0, d12
	fsub	d10, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d10
	fmov	d0, d12
	bl	cos
	fmov	d10, #1.00000000
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.3:
	fmov	d0, d12
	fsub	d11, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d11
	fmov	d0, d12
	bl	cos
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.4:
	fmov	d0, d12
	fsub	d10, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d10
	fmov	d0, d12
	bl	cos
	fmov	d10, #1.00000000
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.5:
	fmov	d0, d12
	fsub	d11, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d11
	fmov	d0, d12
	bl	cos
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.6:
	fmov	d0, d12
	fsub	d10, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d10
	fmov	d0, d12
	bl	cos
	fmov	d10, #1.00000000
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.7:
	fmov	d0, d12
	fsub	d11, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d11
	fmov	d0, d12
	bl	cos
	ldr	d1, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d1, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.8:
	fmov	d0, d12
	fsub	d10, d14, d12
	bl	sin
	fmadd	d11, d15, d0, d10
	fmov	d0, d12
	bl	cos
	fmov	d10, #1.00000000
	ldr	d2, [x29, #24]                  // 8-byte Folded Reload
	fmadd	d0, d2, d0, d10
	fdiv	d0, d11, d0
	fabs	d1, d0
	fadd	d12, d12, d0
	fcmp	d1, d9
	b.mi	.LBB1_10
// %bb.9:
	fmov	d0, d12
	fsub	d9, d14, d12
	fmov	d11, d2
	bl	sin
	fmadd	d9, d15, d0, d9
	fmov	d0, d12
	bl	cos
	fmadd	d0, d11, d0, d10
	fdiv	d0, d9, d0
	fadd	d12, d12, d0
.LBB1_10:
	fmov	d5, #1.00000000
.LBB1_11:
	ldp	d1, d0, [sp]                    // 16-byte Folded Reload
	mov	w8, #8                          // =0x8
	ldr	d2, [sp, #16]                   // 8-byte Folded Reload
	ldr	d7, [sp, #232]                  // 8-byte Folded Reload
	fmul	d1, d0, d1
	ldr	d0, [sp, #32]                   // 8-byte Folded Reload
	fmul	d2, d0, d2
	ldp	d6, d0, [x29, #-152]            // 16-byte Folded Reload
	fcmp	d0, #0.0
	ldp	d3, d0, [x29, #-200]            // 16-byte Folded Reload
	fmul	d3, d0, d3
	ldp	d4, d0, [x29, #-128]            // 16-byte Folded Reload
	csel	x8, x8, xzr, mi
	stur	d15, [x29, #-120]               // 8-byte Folded Spill
	fmadd	d4, d4, d8, d0
	fadd	d0, d15, d5
	fsub	d5, d5, d15
	fdiv	d0, d0, d5
	ldur	d5, [x29, #-136]                // 8-byte Folded Reload
	fmadd	d3, d5, d6, d3
	ldp	d6, d5, [x29, #-184]            // 16-byte Folded Reload
	fmul	d5, d5, d6
	ldur	d6, [x29, #-88]                 // 8-byte Folded Reload
	fmadd	d4, d4, d8, d6
	ldur	d6, [x29, #-224]                // 8-byte Folded Reload
	fmul	d6, d6, d7
	ldp	d16, d7, [x29, #-168]           // 16-byte Folded Reload
	fmadd	d3, d3, d13, d4
	fmadd	d5, d7, d16, d5
	ldp	d7, d4, [x29, #-216]            // 16-byte Folded Reload
	fmadd	d4, d4, d7, d6
	ldp	d7, d6, [sp, #200]              // 16-byte Folded Reload
	fmadd	d3, d5, d13, d3
	fmul	d6, d6, d7
	ldp	d7, d5, [sp, #168]              // 16-byte Folded Reload
	fmadd	d3, d4, d13, d3
	ldr	d4, [sp, #144]                  // 8-byte Folded Reload
	fmul	d5, d5, d7
	ldp	d16, d7, [sp, #216]             // 16-byte Folded Reload
	fsqrt	d9, d0
	fmadd	d6, d7, d16, d6
	ldr	d7, [sp, #128]                  // 8-byte Folded Reload
	fmul	d4, d4, d7
	ldp	d16, d7, [sp, #184]             // 16-byte Folded Reload
	fmadd	d3, d6, d13, d3
	fmadd	d5, d7, d16, d5
	ldp	d7, d6, [sp, #152]              // 16-byte Folded Reload
	fmadd	d4, d6, d7, d4
	fmadd	d3, d5, d13, d3
	ldp	d6, d5, [sp, #112]              // 16-byte Folded Reload
	fmadd	d2, d5, d6, d2
	ldr	d5, [sp, #24]                   // 8-byte Folded Reload
	ldr	d6, [sp, #88]                   // 8-byte Folded Reload
	fmadd	d3, d4, d13, d3
	ldr	d4, [sp, #48]                   // 8-byte Folded Reload
	fmul	d4, d4, d5
	ldr	d5, [sp, #104]                  // 8-byte Folded Reload
	fmadd	d1, d5, d6, d1
	fmadd	d2, d2, d13, d3
	ldur	d6, [x29, #-112]                // 8-byte Folded Reload
	ldp	d5, d3, [sp, #56]               // 16-byte Folded Reload
	fmadd	d3, d3, d5, d4
	ldp	d5, d7, [x29, #-104]            // 16-byte Folded Reload
	fmadd	d1, d1, d13, d2
	fabs	d4, d7
	fmadd	d5, d6, d8, d5
	ldr	d6, [x20, x8]
	fmul	d3, d8, d3
	mov	x8, #35184372088832             // =0x200000000000
	fsub	d2, d7, d6
	ldr	d6, [sp, #96]                   // 8-byte Folded Reload
	movk	x8, #16556, lsl #48
	fcmp	d4, d6
	fmul	d4, d8, d5
	fmov	d5, #0.50000000
	fmadd	d14, d3, d13, d1
	fmov	d1, x8
	fcsel	d8, d7, d2, lt
	ldur	d2, [x29, #-80]                 // 8-byte Folded Reload
	fmul	d10, d12, d5
	fcmp	d9, d9
	fmadd	d11, d2, d1, d4
	b.vs	.LBB1_15
.LBB1_12:                               // %.split
	ldur	d0, [x29, #-72]                 // 8-byte Folded Reload
	fmul	d11, d11, d0
	fmov	d0, d10
	bl	sin
	fmul	d9, d9, d0
	fmov	d0, d10
	bl	cos
	fmov	d1, d0
	fmov	d0, d9
	bl	atan2
	fadd	d15, d0, d0
	fmov	d0, d12
	bl	cos
	fmov	d3, #1.00000000
	adrp	x8, amas
	add	x8, x8, :lo12:amas
	fmov	d2, d0
	ldr	d0, [x8, x21, lsl #3]
	fmul	d1, d14, d14
	ldr	d12, [x29, #24]                 // 8-byte Folded Reload
	ldr	d13, [sp, #136]                 // 8-byte Folded Reload
	fdiv	d0, d3, d0
	fmul	d1, d14, d1
	fmadd	d9, d12, d2, d3
	fadd	d0, d0, d3
	fdiv	d1, d0, d1
	fsqrt	d0, d1
	fcmp	d0, d0
	b.vs	.LBB1_16
.LBB1_13:                               // %.split.split
	adrp	x8, .LCPI1_8
	fmov	d1, #0.50000000
	fmul	d3, d14, d9
	ldr	d2, [x8, :lo12:.LCPI1_8]
	fmul	d0, d0, d2
	fmul	d10, d11, d1
	stp	d3, d0, [x29, #-80]             // 16-byte Folded Spill
	fmov	d0, d10
	bl	sin
	fmov	d9, d0
	fmov	d0, d8
	bl	cos
	fmul	d11, d9, d0
	fmov	d0, d8
	bl	sin
	fadd	d8, d13, d15
	fmul	d15, d9, d0
	fmov	d0, d8
	bl	sin
	fmov	d9, d0
	fmov	d0, d8
	bl	cos
	fmov	d3, d0
	fmov	d0, #1.00000000
	ldur	d8, [x29, #-120]                // 8-byte Folded Reload
	fnmul	d2, d9, d11
	fmadd	d1, d12, d8, d0
	str	d3, [x29, #24]                  // 8-byte Folded Spill
	fmadd	d2, d15, d3, d2
	fsqrt	d0, d1
	fadd	d12, d2, d2
	fcmp	d0, d0
	b.vs	.LBB1_17
.LBB1_14:                               // %.split.split.split
	fdiv	d14, d14, d0
	fmov	d0, d10
	bl	cos
	fmov	d10, d0
	fmov	d0, d13
	bl	sin
	fmadd	d1, d8, d0, d9
	fmov	d0, d13
	fmul	d13, d14, d1
	bl	cos
	ldr	d19, [x29, #24]                 // 8-byte Folded Reload
	fadd	d1, d15, d15
	fmov	d2, #-2.00000000
	fadd	d6, d10, d10
	fmov	d5, #1.00000000
	fnmul	d7, d12, d10
	fmadd	d0, d8, d0, d19
	ldp	d18, d17, [x29, #-80]           // 16-byte Folded Reload
	fmul	d2, d11, d2
	fmadd	d16, d12, d11, d9
	adrp	x8, .LCPI1_9
	fmul	d4, d11, d1
	fmul	d7, d18, d7
	fmul	d0, d14, d0
	fmadd	d2, d2, d11, d5
	fmov	d5, #-1.00000000
	fmul	d3, d11, d0
	fmadd	d1, d1, d15, d5
	fmsub	d5, d12, d15, d19
	fmadd	d3, d15, d13, d3
	fmul	d3, d6, d3
	fnmul	d6, d13, d4
	fmul	d4, d4, d0
	fmadd	d0, d2, d0, d6
	fmul	d3, d17, d3
	ldr	q2, [x8, :lo12:.LCPI1_9]
	fmul	d6, d18, d16
	fmadd	d1, d1, d13, d4
	adrp	x8, .LCPI1_10
	fmul	v7.2d, v2.2d, v7.d[0]
	ldr	q4, [x8, :lo12:.LCPI1_10]
	fmul	d0, d17, d0
	fmul	v2.2d, v2.2d, v3.d[0]
	fmul	d3, d18, d5
	fmul	d1, d17, d1
	fmla	v7.2d, v4.2d, v6.d[0]
	fmla	v2.2d, v4.2d, v0.d[0]
	str	d3, [x19]
	str	d1, [x19, #24]
	stur	q7, [x19, #8]
	str	q2, [x19, #32]
	add	sp, sp, #400
	.cfi_def_cfa wsp, 160
	ldp	x20, x19, [sp, #144]            // 16-byte Folded Reload
	ldr	x28, [sp, #80]                  // 8-byte Folded Reload
	ldp	x22, x21, [sp, #128]            // 16-byte Folded Reload
	ldp	x24, x23, [sp, #112]            // 16-byte Folded Reload
	ldp	x26, x25, [sp, #96]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #64]             // 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               // 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             // 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             // 16-byte Folded Reload
	ldp	d15, d14, [sp], #160            // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w23
	.cfi_restore w24
	.cfi_restore w25
	.cfi_restore w26
	.cfi_restore w28
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
.LBB1_15:                               // %call.sqrt
	.cfi_restore_state
	bl	sqrt
	fmov	d9, d0
	b	.LBB1_12
.LBB1_16:                               // %call.sqrt1
	fmov	d0, d1
	bl	sqrt
	b	.LBB1_13
.LBB1_17:                               // %call.sqrt2
	fmov	d0, d1
	bl	sqrt
	b	.LBB1_14
.Lfunc_end1:
	.size	planetpv, .Lfunc_end1-planetpv
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function radecdist
.LCPI2_0:
	.xword	0x400e8ec8a4aeacc4              // double 3.8197186342054881
.LCPI2_1:
	.xword	0x404ca5dc1a63c1f8              // double 57.295779513082323
	.text
	.globl	radecdist
	.p2align	2
	.type	radecdist,@function
radecdist:                              // @radecdist
	.cfi_startproc
// %bb.0:
	str	d8, [sp, #-48]!                 // 8-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 32
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w30, -24
	.cfi_offset w29, -32
	.cfi_offset b8, -48
	ldp	d1, d0, [x0]
	mov	x19, x1
	mov	x20, x0
	fmul	d0, d0, d0
	fmadd	d0, d1, d1, d0
	ldr	d1, [x0, #16]
	fmadd	d0, d1, d1, d0
	fsqrt	d8, d0
	str	d8, [x1, #16]
	ldp	d1, d0, [x0]
	bl	atan2
	adrp	x8, .LCPI2_0
	ldr	d1, [x8, :lo12:.LCPI2_0]
	fmul	d0, d0, d1
	fmov	d1, #24.00000000
	fadd	d1, d0, d1
	fcmp	d0, #0.0
	fcsel	d0, d1, d0, mi
	str	d0, [x19]
	ldr	d0, [x20, #16]
	fdiv	d0, d0, d8
	bl	asin
	adrp	x8, .LCPI2_1
	ldr	d1, [x8, :lo12:.LCPI2_1]
	fmul	d0, d0, d1
	str	d0, [x19, #8]
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	ldr	d8, [sp], #48                   // 8-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w30
	.cfi_restore w29
	.cfi_restore b8
	ret
.Lfunc_end2:
	.size	radecdist, .Lfunc_end2-radecdist
	.cfi_endproc
                                        // -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          // -- Begin function main
.LCPI3_0:
	.xword	0x4142b42c80000000              // double 2451545
.LCPI3_1:
	.xword	0x404ca5dc1a63c1f8              // double 57.295779513082323
.LCPI3_2:
	.xword	0x400e8ec8a4aeacc4              // double 3.8197186342054881
	.text
	.globl	main
	.p2align	2
	.type	main,@function
main:                                   // @main
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #320
	.cfi_def_cfa_offset 320
	stp	d15, d14, [sp, #208]            // 16-byte Folded Spill
	stp	d13, d12, [sp, #224]            // 16-byte Folded Spill
	stp	d11, d10, [sp, #240]            // 16-byte Folded Spill
	stp	d9, d8, [sp, #256]              // 16-byte Folded Spill
	stp	x29, x30, [sp, #272]            // 16-byte Folded Spill
	str	x28, [sp, #288]                 // 8-byte Folded Spill
	stp	x20, x19, [sp, #304]            // 16-byte Folded Spill
	add	x29, sp, #272
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w28, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	.cfi_offset b8, -56
	.cfi_offset b9, -64
	.cfi_offset b10, -72
	.cfi_offset b11, -80
	.cfi_offset b12, -88
	.cfi_offset b13, -96
	.cfi_offset b14, -104
	.cfi_offset b15, -112
	adrp	x8, .LCPI3_0
	fmov	d1, #1.00000000
	mov	w19, wzr
	ldr	d0, [x8, :lo12:.LCPI3_0]
	str	d0, [sp, #16]                   // 8-byte Folded Spill
.LBB3_1:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB3_2 Depth 2
	ldr	d2, [sp, #16]                   // 8-byte Folded Reload
	mov	w20, #36525                     // =0x8ead
	stur	xzr, [x29, #-72]
.LBB3_2:                                //   Parent Loop BB3_1 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	fadd	d2, d2, d1
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, wzr
	str	d2, [x29, #24]                  // 8-byte Folded Spill
	stur	d2, [x29, #-80]
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d8, d9
	stp	d0, d9, [sp, #128]              // 16-byte Folded Spill
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #1                          // =0x1
	str	d0, [sp, #120]                  // 8-byte Folded Spill
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d8, d9
	stp	d0, d9, [sp, #104]              // 16-byte Folded Spill
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #2                          // =0x2
	str	d0, [sp, #96]                   // 8-byte Folded Spill
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d8, d9
	str	d0, [sp, #88]                   // 8-byte Folded Spill
	str	d9, [sp, #40]                   // 8-byte Folded Spill
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #3                          // =0x3
	str	d0, [sp, #80]                   // 8-byte Folded Spill
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d8, d9
	str	d0, [sp, #72]                   // 8-byte Folded Spill
	str	d9, [sp, #32]                   // 8-byte Folded Spill
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #4                          // =0x4
	str	d0, [sp, #64]                   // 8-byte Folded Spill
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d8, d9
	str	d0, [sp, #56]                   // 8-byte Folded Spill
	str	d9, [sp, #24]                   // 8-byte Folded Spill
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #5                          // =0x5
	str	d0, [sp, #48]                   // 8-byte Folded Spill
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d8, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d8, d8, d2
	fsqrt	d13, d2
	bl	atan2
	fdiv	d1, d8, d13
	fmov	d10, d0
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #6                          // =0x6
	fmov	d14, d0
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d9, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d9, d9, d2
	fsqrt	d8, d2
	bl	atan2
	fdiv	d1, d9, d8
	fmov	d11, d0
	fmov	d0, d1
	bl	asin
	sub	x0, x29, #80
	sub	x2, x29, #128
	mov	w1, #7                          // =0x7
	fmov	d12, d0
	bl	planetpv
	ldp	d1, d0, [x29, #-128]
	ldur	d15, [x29, #-112]
	fmul	d2, d0, d0
	fmadd	d2, d1, d1, d2
	fmadd	d2, d15, d15, d2
	fsqrt	d9, d2
	bl	atan2
	fdiv	d1, d15, d9
	fmov	d15, d0
	fmov	d0, d1
	bl	asin
	fmov	d1, #1.00000000
	ldr	d2, [x29, #24]                  // 8-byte Folded Reload
	subs	w20, w20, #1
	b.ne	.LBB3_2
// %bb.3:                               //   in Loop: Header=BB3_1 Depth=1
	fmov	d28, d12
	ldp	d24, d27, [sp, #48]             // 16-byte Folded Reload
	ldp	d23, d25, [sp, #64]             // 16-byte Folded Reload
	add	w19, w19, #1
	ldp	d21, d22, [sp, #80]             // 16-byte Folded Reload
	cmp	w19, #20
	ldp	d19, d20, [sp, #96]             // 16-byte Folded Reload
	ldr	d12, [sp, #112]                 // 8-byte Folded Reload
	ldp	d17, d18, [sp, #120]            // 16-byte Folded Reload
	ldr	d2, [sp, #136]                  // 8-byte Folded Reload
	stp	d9, d8, [sp]                    // 16-byte Folded Spill
	str	d13, [x29, #24]                 // 8-byte Folded Spill
	b.ne	.LBB3_1
// %bb.4:
	adrp	x8, .LCPI3_2
	fmov	d3, #24.00000000
	fmov	d26, d14
	ldr	d1, [x8, :lo12:.LCPI3_2]
	adrp	x8, .LCPI3_1
	adrp	x19, .L.str.1
	add	x19, x19, :lo12:.L.str.1
	fmul	d16, d15, d1
	fmul	d4, d11, d1
	mov	x0, x19
	fmul	d6, d10, d1
	fadd	d5, d16, d3
	fcmp	d16, #0.0
	fadd	d7, d4, d3
	fcsel	d8, d5, d16, mi
	fcmp	d4, #0.0
	fmul	d16, d27, d1
	fadd	d5, d6, d3
	fcsel	d9, d7, d4, mi
	fcmp	d6, #0.0
	fmul	d4, d25, d1
	fadd	d7, d16, d3
	fcsel	d5, d5, d6, mi
	fcmp	d16, #0.0
	fadd	d6, d4, d3
	fcsel	d7, d7, d16, mi
	fcmp	d4, #0.0
	str	d5, [sp, #72]                   // 8-byte Folded Spill
	fmul	d5, d22, d1
	fmul	d16, d20, d1
	fcsel	d4, d6, d4, mi
	str	d7, [sp, #88]                   // 8-byte Folded Spill
	fadd	d7, d5, d3
	fcmp	d5, #0.0
	fadd	d6, d16, d3
	str	d4, [sp, #16]                   // 8-byte Folded Spill
	fmul	d4, d18, d1
	ldr	d1, [x8, :lo12:.LCPI3_1]
	fmul	d0, d0, d1
	fmul	d10, d23, d1
	fcsel	d14, d7, d5, mi
	fcmp	d16, #0.0
	fmul	d5, d26, d1
	fmul	d13, d21, d1
	fmul	d15, d19, d1
	str	d0, [sp, #128]                  // 8-byte Folded Spill
	fmul	d0, d28, d1
	fcsel	d11, d6, d16, mi
	fcmp	d4, #0.0
	str	d0, [sp, #104]                  // 8-byte Folded Spill
	fadd	d0, d4, d3
	fmul	d3, d24, d1
	fmul	d1, d17, d1
	fcsel	d0, d0, d4, mi
	stp	d3, d5, [sp, #48]               // 16-byte Folded Spill
	bl	printf
	fmov	d0, d11
	fmov	d1, d15
	mov	x0, x19
	fmov	d2, d12
	bl	printf
	fmov	d0, d14
	fmov	d1, d13
	ldr	d2, [sp, #40]                   // 8-byte Folded Reload
	mov	x0, x19
	bl	printf
	fmov	d1, d10
	ldr	d0, [sp, #16]                   // 8-byte Folded Reload
	ldr	d2, [sp, #32]                   // 8-byte Folded Reload
	mov	x0, x19
	bl	printf
	ldr	d0, [sp, #88]                   // 8-byte Folded Reload
	ldr	d1, [sp, #48]                   // 8-byte Folded Reload
	mov	x0, x19
	ldr	d2, [sp, #24]                   // 8-byte Folded Reload
	bl	printf
	ldr	d0, [sp, #72]                   // 8-byte Folded Reload
	ldr	d1, [sp, #56]                   // 8-byte Folded Reload
	mov	x0, x19
	ldr	d2, [x29, #24]                  // 8-byte Folded Reload
	bl	printf
	fmov	d0, d9
	ldr	d1, [sp, #104]                  // 8-byte Folded Reload
	ldr	d2, [sp, #8]                    // 8-byte Folded Reload
	mov	x0, x19
	bl	printf
	fmov	d0, d8
	ldr	d1, [sp, #128]                  // 8-byte Folded Reload
	ldr	d2, [sp]                        // 8-byte Folded Reload
	mov	x0, x19
	bl	printf
	adrp	x8, :got:stdout
	ldr	x8, [x8, :got_lo12:stdout]
	ldr	x0, [x8]
	bl	fflush
	mov	w0, wzr
	.cfi_def_cfa wsp, 320
	ldp	x20, x19, [sp, #304]            // 16-byte Folded Reload
	ldr	x28, [sp, #288]                 // 8-byte Folded Reload
	ldp	x29, x30, [sp, #272]            // 16-byte Folded Reload
	ldp	d9, d8, [sp, #256]              // 16-byte Folded Reload
	ldp	d11, d10, [sp, #240]            // 16-byte Folded Reload
	ldp	d13, d12, [sp, #224]            // 16-byte Folded Reload
	ldp	d15, d14, [sp, #208]            // 16-byte Folded Reload
	add	sp, sp, #320
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w28
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
.Lfunc_end3:
	.size	main, .Lfunc_end3-main
	.cfi_endproc
                                        // -- End function
	.type	a,@object                       // @a
	.section	.rodata,"a",@progbits
	.p2align	3, 0x0
a:
	.xword	0x3fd8c637fd3b6253              // double 0.38709830979999998
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x3fe725849423e3e0              // double 0.72332982000000001
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x3ff000011136aef5              // double 1.0000010178000001
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x3ff860fd96f0d223              // double 1.5236793419000001
	.xword	0x3df49da7e361ce4c              // double 3.0E-10
	.xword	0x0000000000000000              // double 0
	.xword	0x4014cf7737365089              // double 5.2026032092000003
	.xword	0x3ec00c90d2b8ae8f              // double 1.9132000000000002E-6
	.xword	0xbe30c01868bf779e              // double -3.9000000000000002E-9
	.xword	0x40231c1d0ebb7c0f              // double 9.5549091915000001
	.xword	0xbef66dba1e9d9a9d              // double -2.1389599999999999E-5
	.xword	0x3e67d64a1ee91688              // double 4.4400000000000001E-8
	.xword	0x403337ec14c35efa              // double 19.218446061800002
	.xword	0xbe98f00a1561f9e1              // double -3.7160000000000002E-7
	.xword	0x3e7a47a3038502a4              // double 9.7899999999999997E-8
	.xword	0x403e1c425059fb17              // double 30.110386869399999
	.xword	0xbebbe8ad97c217d7              // double -1.6635E-6
	.xword	0x3e726a27f670079b              // double 6.8600000000000005E-8
	.size	a, 192

	.type	dlm,@object                     // @dlm
	.p2align	3, 0x0
dlm:
	.xword	0x406f88076b035926              // double 252.25090552
	.xword	0x41f40bbcadee3cb4              // double 5381016286.8898201
	.xword	0xbffed8a32f44912a              // double -1.9278900000000001
	.xword	0x4066bf5a874feafa              // double 181.97980085
	.xword	0x41df6432f5157881              // double 2106641364.33548
	.xword	0x3fe3007dd4413554              // double 0.59380999999999995
	.xword	0x40591dda6dbf7622              // double 100.46645683
	.xword	0x41d34fc2f3b56502              // double 1295977422.83429
	.xword	0xc0005a5657fb6998              // double -2.0441099999999999
	.xword	0x407636ed90f7b482              // double 355.43299958
	.xword	0x41c4890a4b784dfd              // double 689050774.93988001
	.xword	0x3fee2a1b5c7cd899              // double 0.94264000000000003
	.xword	0x40412cfe90ea1d96              // double 34.351518740000003
	.xword	0x419a0c7e6f1ea0ba              // double 109256603.77991
	.xword	0xc03e9a915379fa98              // double -30.60378
	.xword	0x404909e9b1dfe17d              // double 50.077444300000003
	.xword	0x4184fa9e14756430              // double 43996098.557319999
	.xword	0x4052e76ed677707a              // double 75.616140000000001
	.xword	0x4073a0e14d09c902              // double 314.05500511000002
	.xword	0x416d6ba57e0efdca              // double 15424811.93933
	.xword	0xbffc0366516db0de              // double -1.7508300000000001
	.xword	0x4073059422411d82              // double 304.34866548000002
	.xword	0x415e0127cd46b26c              // double 7865503.20744
	.xword	0x3fcb0307f23cc8de              // double 0.21103
	.size	dlm, 192

	.type	e,@object                       // @e
	.p2align	3, 0x0
e:
	.xword	0x3fca52242a37d430              // double 0.2056317526
	.xword	0x3f2abf4b9459e7f4              // double 2.0406530000000001E-4
	.xword	0xbec7c7e6c1bd0f9b              // double -2.8349000000000002E-6
	.xword	0x3f7bbcde77820827              // double 0.0067719164000000004
	.xword	0xbf3f4dac25fb4bc2              // double -4.7765209999999999E-4
	.xword	0x3ee4942737feff37              // double 9.8127000000000001E-6
	.xword	0x3f911c1175cc9f7b              // double 0.016708634199999999
	.xword	0xbf3b8c8fa536f731              // double -4.2036539999999999E-4
	.xword	0xbeea93fad53b08d4              // double -1.26734E-5
	.xword	0x3fb7e91ad74bf5b0              // double 0.093400647700000005
	.xword	0x3f4da66143b5e407              // double 9.0484379999999995E-4
	.xword	0xbee0e96176f62826              // double -8.0640999999999996E-6
	.xword	0x3fa8d4b857e48742              // double 0.048497925499999997
	.xword	0x3f5abe2b9a18b7b5              // double 0.0016322541999999999
	.xword	0xbf08b6913e59c18d              // double -4.7136599999999999E-5
	.xword	0x3fac70ce5fa41e66              // double 0.055548142600000003
	.xword	0xbf6c6594a86fd58e              // double -0.0034664062000000001
	.xword	0xbf10df6361d60729              // double -6.4363900000000002E-5
	.xword	0x3fa7bf479022d287              // double 0.046381222100000001
	.xword	0xbf31e2fe6ae927d8              // double -2.7292929999999998E-4
	.xword	0x3ee08c9c0376f006              // double 7.8913000000000002E-6
	.xword	0x3f835d88e0fe76d8              // double 0.0094557470000000004
	.xword	0x3f0fa0dbe27c5204              // double 6.0326299999999999E-5
	.xword	0x0000000000000000              // double 0
	.size	e, 192

	.type	pi,@object                      // @pi
	.p2align	3, 0x0
pi:
	.xword	0x40535d310de9f882              // double 77.456119040000004
	.xword	0x40b6571dab9f559b              // double 5719.1158999999998
	.xword	0xc01352157689ca19              // double -4.8301600000000002
	.xword	0x40607209dadfb507              // double 131.563703
	.xword	0x4065ef9096bb98c8              // double 175.4864
	.xword	0xc07f27b59ddc1e79              // double -498.48183999999998
	.xword	0x4059bbfd82cd2461              // double 102.93734808000001
	.xword	0x40c6ae2d2bd3c361              // double 11612.3529
	.xword	0x404aa34c6e6d9be5              // double 53.275770000000001
	.xword	0x407500f6b7dfd5be              // double 336.06023395
	.xword	0x40cf363ac3222920              // double 15980.459080000001
	.xword	0xc04f29fbe76c8b44              // double -62.328000000000003
	.xword	0x402ca993f265b897              // double 14.331206870000001
	.xword	0x40be4ec06ad2dcb1              // double 7758.7516299999997
	.xword	0x40703f599ed7c6fc              // double 259.95938000000001
	.xword	0x405743a9c7642d26              // double 93.057237479999997
	.xword	0x40d3eadfa415f45e              // double 20395.49439
	.xword	0x4067c84dfce3150e              // double 190.25952000000001
	.xword	0x4065a02b58283528              // double 173.00529105999999
	.xword	0x40a91f1ff04577d9              // double 3215.5623799999998
	.xword	0xc0410be37de939eb              // double -34.092880000000001
	.xword	0x40480f65305b6785              // double 48.120275540000002
	.xword	0x40906ae060fe4799              // double 1050.71912
	.xword	0x403b65aceee0f3cb              // double 27.397169999999999
	.size	pi, 192

	.type	dinc,@object                    // @dinc
	.p2align	3, 0x0
dinc:
	.xword	0x401c051b1d92b7fe              // double 7.00498625
	.xword	0xc06ac83387160957              // double -214.25629000000001
	.xword	0x3fd28b97785729b3              // double 0.28977000000000003
	.xword	0x400b28447e34386c              // double 3.3946618900000001
	.xword	0xc03ed828a1dfb939              // double -30.844370000000001
	.xword	0xc0275b52007dd441              // double -11.67836
	.xword	0x0000000000000000              // double 0
	.xword	0x407d5f90f51ac9b0              // double 469.97289000000001
	.xword	0xc00acde2ac322292              // double -3.35053
	.xword	0x3ffd987acb2252bb              // double 1.84972648
	.xword	0xc072551355475a32              // double -293.31722000000002
	.xword	0xc0203c91d14e3bcd              // double -8.1182999999999996
	.xword	0x3ff4da2e7a10e830              // double 1.3032669800000001
	.xword	0xc051e3c504816f00              // double -71.558899999999994
	.xword	0x4027e7ebaf102364              // double 11.952970000000001
	.xword	0x4003e939471e778f              // double 2.4888787799999998
	.xword	0x4056f686594af4f1              // double 91.851950000000002
	.xword	0xc031a989374bc6a8              // double -17.66225
	.xword	0x3fe8be07677d67b5              // double 0.77319689000000003
	.xword	0xc04e5d15df6555c5              // double -60.727229999999999
	.xword	0x3ff41f16b11c6d1e              // double 1.25759
	.xword	0x3ffc51b9ce9853f4              // double 1.7699525899999999
	.xword	0x40203f251c193b3a              // double 8.1233299999999993
	.xword	0x3fb4d35a858793de              // double 0.081350000000000005
	.size	dinc, 192

	.type	omega,@object                   // @omega
	.p2align	3, 0x0
omega:
	.xword	0x40482a5ab400a313              // double 48.330893039999999
	.xword	0xc0b1a3379f01b867              // double -4515.2172700000001
	.xword	0xc03fcc8605681ecd              // double -31.798919999999999
	.xword	0x40532b83cff8fc2b              // double 76.679920190000004
	.xword	0xc0c38c3da31a4bdc              // double -10008.481540000001
	.xword	0xc049a9bef49cf56f              // double -51.326140000000002
	.xword	0x4065dbf10e4ff9e8              // double 174.87317576999999
	.xword	0xc0c0f3a29a804966              // double -8679.2703399999991
	.xword	0x402eaf0ed3d859c9              // double 15.34191
	.xword	0x4048c76f992a88eb              // double 49.558093210000003
	.xword	0xc0c4be7350092ccf              // double -10620.900879999999
	.xword	0xc06cd25f84cad57c              // double -230.57416000000001
	.xword	0x40591db8d838bbb3              // double 100.46440702
	.xword	0x40b8da091dbca969              // double 6362.0356099999999
	.xword	0x4074685935fc3b4f              // double 326.52177999999998
	.xword	0x405c6a9797e1b38f              // double 113.66550252
	.xword	0xc0c20c1986983516              // double -9240.1994200000008
	.xword	0xc0508f320d9945b7              // double -66.237430000000003
	.xword	0x405280619982c872              // double 74.005957010000003
	.xword	0x40a4da4cf80dc337              // double 2669.1503299999999
	.xword	0x40623e1187e7c06e              // double 145.93964
	.xword	0x40607916febf632d              // double 131.78405702000001
	.xword	0xc06bbe2edbb59ddc              // double -221.94322
	.xword	0xbfe93165d3996fa8              // double -0.78727999999999998
	.size	omega, 192

	.type	kp,@object                      // @kp
	.p2align	3, 0x0
kp:
	.xword	0x40f0fed000000000              // double 69613
	.xword	0x40f277d000000000              // double 75645
	.xword	0x40f58f2000000000              // double 88306
	.xword	0x40ed3f6000000000              // double 59899
	.xword	0x40cec10000000000              // double 15746
	.xword	0x40f15af000000000              // double 71087
	.xword	0x41015ae800000000              // double 142173
	.xword	0x40a81c0000000000              // double 3086
	.xword	0x0000000000000000              // double 0
	.xword	0x40d559c000000000              // double 21863
	.xword	0x40e0034000000000              // double 32794
	.xword	0x40da4d8000000000              // double 26934
	.xword	0x40c5598000000000              // double 10931
	.xword	0x40d9a28000000000              // double 26250
	.xword	0x40e559a000000000              // double 43725
	.xword	0x40ea4d6000000000              // double 53867
	.xword	0x40dc42c000000000              // double 28939
	.xword	0x0000000000000000              // double 0
	.xword	0x40cf410000000000              // double 16002
	.xword	0x40d559c000000000              // double 21863
	.xword	0x40df410000000000              // double 32004
	.xword	0x40c5598000000000              // double 10931
	.xword	0x40cc608000000000              // double 14529
	.xword	0x40cff80000000000              // double 16368
	.xword	0x40cdeb0000000000              // double 15318
	.xword	0x40e0034000000000              // double 32794
	.xword	0x0000000000000000              // double 0
	.xword	0x40b8c90000000000              // double 6345
	.xword	0x40be8a0000000000              // double 7818
	.xword	0x40ce8a0000000000              // double 15636
	.xword	0x40bba50000000000              // double 7077
	.xword	0x40bff80000000000              // double 8184
	.xword	0x40cba98000000000              // double 14163
	.xword	0x40914c0000000000              // double 1107
	.xword	0x40b3080000000000              // double 4872
	.xword	0x0000000000000000              // double 0
	.xword	0x409b800000000000              // double 1760
	.xword	0x4096b80000000000              // double 1454
	.xword	0x40923c0000000000              // double 1167
	.xword	0x408b800000000000              // double 880
	.xword	0x4071f00000000000              // double 287
	.xword	0x40a4a00000000000              // double 2640
	.xword	0x4033000000000000              // double 19
	.xword	0x409ffc0000000000              // double 2047
	.xword	0x4096b80000000000              // double 1454
	.xword	0x4081f00000000000              // double 574
	.xword	0x0000000000000000              // double 0
	.xword	0x408b800000000000              // double 880
	.xword	0x4071f00000000000              // double 287
	.xword	0x4033000000000000              // double 19
	.xword	0x409b800000000000              // double 1760
	.xword	0x40923c0000000000              // double 1167
	.xword	0x4073200000000000              // double 306
	.xword	0x4081f00000000000              // double 574
	.xword	0x4069800000000000              // double 204
	.xword	0x0000000000000000              // double 0
	.xword	0x4066200000000000              // double 177
	.xword	0x4093c40000000000              // double 1265
	.xword	0x4010000000000000              // double 4
	.xword	0x4078100000000000              // double 385
	.xword	0x4069000000000000              // double 200
	.xword	0x406a000000000000              // double 208
	.xword	0x4069800000000000              // double 204
	.xword	0x0000000000000000              // double 0
	.xword	0x4059800000000000              // double 102
	.xword	0x405a800000000000              // double 106
	.xword	0x4010000000000000              // double 4
	.xword	0x4058800000000000              // double 98
	.xword	0x40955c0000000000              // double 1367
	.xword	0x407e700000000000              // double 487
	.xword	0x4069800000000000              // double 204
	.xword	0x0000000000000000              // double 0
	.size	kp, 576

	.type	kq,@object                      // @kq
	.p2align	3, 0x0
kq:
	.xword	0x40a81c0000000000              // double 3086
	.xword	0x40cec10000000000              // double 15746
	.xword	0x40f0fed000000000              // double 69613
	.xword	0x40ed3f6000000000              // double 59899
	.xword	0x40f277d000000000              // double 75645
	.xword	0x40f58f2000000000              // double 88306
	.xword	0x40c8ba8000000000              // double 12661
	.xword	0x40a4c40000000000              // double 2658
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x40d559c000000000              // double 21863
	.xword	0x40e0034000000000              // double 32794
	.xword	0x40c5598000000000              // double 10931
	.xword	0x4052400000000000              // double 73
	.xword	0x40b1230000000000              // double 4387
	.xword	0x40da4d8000000000              // double 26934
	.xword	0x4097040000000000              // double 1473
	.xword	0x40a0da0000000000              // double 2157
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x4024000000000000              // double 10
	.xword	0x40cf410000000000              // double 16002
	.xword	0x40d559c000000000              // double 21863
	.xword	0x40c5598000000000              // double 10931
	.xword	0x4097040000000000              // double 1473
	.xword	0x40df410000000000              // double 32004
	.xword	0x40b1230000000000              // double 4387
	.xword	0x4052400000000000              // double 73
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x4024000000000000              // double 10
	.xword	0x40b8c90000000000              // double 6345
	.xword	0x40be8a0000000000              // double 7818
	.xword	0x40914c0000000000              // double 1107
	.xword	0x40ce8a0000000000              // double 15636
	.xword	0x40bba50000000000              // double 7077
	.xword	0x40bff80000000000              // double 8184
	.xword	0x4080a00000000000              // double 532
	.xword	0x4024000000000000              // double 10
	.xword	0x0000000000000000              // double 0
	.xword	0x4033000000000000              // double 19
	.xword	0x409b800000000000              // double 1760
	.xword	0x4096b80000000000              // double 1454
	.xword	0x4071f00000000000              // double 287
	.xword	0x40923c0000000000              // double 1167
	.xword	0x408b800000000000              // double 880
	.xword	0x4081f00000000000              // double 574
	.xword	0x40a4a00000000000              // double 2640
	.xword	0x4033000000000000              // double 19
	.xword	0x4096b80000000000              // double 1454
	.xword	0x4033000000000000              // double 19
	.xword	0x4081f00000000000              // double 574
	.xword	0x4071f00000000000              // double 287
	.xword	0x4073200000000000              // double 306
	.xword	0x409b800000000000              // double 1760
	.xword	0x4028000000000000              // double 12
	.xword	0x403f000000000000              // double 31
	.xword	0x4043000000000000              // double 38
	.xword	0x4033000000000000              // double 19
	.xword	0x4081f00000000000              // double 574
	.xword	0x4010000000000000              // double 4
	.xword	0x4069800000000000              // double 204
	.xword	0x4066200000000000              // double 177
	.xword	0x4020000000000000              // double 8
	.xword	0x403f000000000000              // double 31
	.xword	0x4069000000000000              // double 200
	.xword	0x4093c40000000000              // double 1265
	.xword	0x4059800000000000              // double 102
	.xword	0x4010000000000000              // double 4
	.xword	0x4069800000000000              // double 204
	.xword	0x4010000000000000              // double 4
	.xword	0x4059800000000000              // double 102
	.xword	0x405a800000000000              // double 106
	.xword	0x4020000000000000              // double 8
	.xword	0x4058800000000000              // double 98
	.xword	0x40955c0000000000              // double 1367
	.xword	0x407e700000000000              // double 487
	.xword	0x4069800000000000              // double 204
	.xword	0x4010000000000000              // double 4
	.xword	0x4059800000000000              // double 102
	.size	kq, 640

	.type	ca,@object                      // @ca
	.p2align	3, 0x0
ca:
	.xword	0x4010000000000000              // double 4
	.xword	0xc02a000000000000              // double -13
	.xword	0x4026000000000000              // double 11
	.xword	0xc022000000000000              // double -9
	.xword	0xc022000000000000              // double -9
	.xword	0xc008000000000000              // double -3
	.xword	0xbff0000000000000              // double -1
	.xword	0x4010000000000000              // double 4
	.xword	0x0000000000000000              // double 0
	.xword	0xc063800000000000              // double -156
	.xword	0x404d800000000000              // double 59
	.xword	0xc045000000000000              // double -42
	.xword	0x4018000000000000              // double 6
	.xword	0x4033000000000000              // double 19
	.xword	0xc034000000000000              // double -20
	.xword	0xc024000000000000              // double -10
	.xword	0xc028000000000000              // double -12
	.xword	0x0000000000000000              // double 0
	.xword	0x4050000000000000              // double 64
	.xword	0xc063000000000000              // double -152
	.xword	0x404f000000000000              // double 62
	.xword	0xc020000000000000              // double -8
	.xword	0x4040000000000000              // double 32
	.xword	0xc044800000000000              // double -41
	.xword	0x4033000000000000              // double 19
	.xword	0xc026000000000000              // double -11
	.xword	0x0000000000000000              // double 0
	.xword	0x405f000000000000              // double 124
	.xword	0x4083680000000000              // double 621
	.xword	0xc062200000000000              // double -145
	.xword	0x406a000000000000              // double 208
	.xword	0x404b000000000000              // double 54
	.xword	0xc04c800000000000              // double -57
	.xword	0x403e000000000000              // double 30
	.xword	0x402e000000000000              // double 15
	.xword	0x0000000000000000              // double 0
	.xword	0xc0d6e34000000000              // double -23437
	.xword	0xc0a4940000000000              // double -2634
	.xword	0x40b9c90000000000              // double 6601
	.xword	0x40b8730000000000              // double 6259
	.xword	0xc0978c0000000000              // double -1507
	.xword	0xc09c740000000000              // double -1821
	.xword	0x40a4780000000000              // double 2620
	.xword	0xc0a0860000000000              // double -2115
	.xword	0xc097440000000000              // double -1489
	.xword	0x40eeb7e000000000              // double 62911
	.xword	0xc0fd46f000000000              // double -119919
	.xword	0x40f35e8000000000              // double 79336
	.xword	0x40d1658000000000              // double 17814
	.xword	0xc0d7ac4000000000              // double -24241
	.xword	0x40c7920000000000              // double 12068
	.xword	0x40c0390000000000              // double 8306
	.xword	0xc0b31d0000000000              // double -4893
	.xword	0x40c1630000000000              // double 8902
	.xword	0x4117bf1400000000              // double 389061
	.xword	0xc10fff6800000000              // double -262125
	.xword	0xc0e5870000000000              // double -44088
	.xword	0x40c0618000000000              // double 8387
	.xword	0xc0d6700000000000              // double -22976
	.xword	0xc0a05a0000000000              // double -2093
	.xword	0xc083380000000000              // double -615
	.xword	0xc0c2fc0000000000              // double -9720
	.xword	0x40b9e90000000000              // double 6633
	.xword	0xc119292c00000000              // double -412235
	.xword	0xc1032bb000000000              // double -157046
	.xword	0xc0deb18000000000              // double -31430
	.xword	0x40e2772000000000              // double 37817
	.xword	0xc0c3060000000000              // double -9740
	.xword	0xc02a000000000000              // double -13
	.xword	0xc0bd190000000000              // double -7449
	.xword	0x40c2d60000000000              // double 9644
	.xword	0x0000000000000000              // double 0
	.size	ca, 576

	.type	sa,@object                      // @sa
	.p2align	3, 0x0
sa:
	.xword	0xc03d000000000000              // double -29
	.xword	0xbff0000000000000              // double -1
	.xword	0x4022000000000000              // double 9
	.xword	0x4018000000000000              // double 6
	.xword	0xc018000000000000              // double -6
	.xword	0x4014000000000000              // double 5
	.xword	0x4010000000000000              // double 4
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0xc048000000000000              // double -48
	.xword	0xc05f400000000000              // double -125
	.xword	0xc03a000000000000              // double -26
	.xword	0xc042800000000000              // double -37
	.xword	0x4032000000000000              // double 18
	.xword	0xc02a000000000000              // double -13
	.xword	0xc034000000000000              // double -20
	.xword	0xc000000000000000              // double -2
	.xword	0x0000000000000000              // double 0
	.xword	0xc062c00000000000              // double -150
	.xword	0xc047000000000000              // double -46
	.xword	0x4051000000000000              // double 68
	.xword	0x404b000000000000              // double 54
	.xword	0x402c000000000000              // double 14
	.xword	0x4038000000000000              // double 24
	.xword	0xc03c000000000000              // double -28
	.xword	0x4036000000000000              // double 22
	.xword	0x0000000000000000              // double 0
	.xword	0xc083680000000000              // double -621
	.xword	0x4080a00000000000              // double 532
	.xword	0xc085b00000000000              // double -694
	.xword	0xc034000000000000              // double -20
	.xword	0x4068000000000000              // double 192
	.xword	0xc057800000000000              // double -94
	.xword	0x4051c00000000000              // double 71
	.xword	0xc052400000000000              // double -73
	.xword	0x0000000000000000              // double 0
	.xword	0xc0cc8b0000000000              // double -14614
	.xword	0xc0d35d0000000000              // double -19828
	.xword	0xc0b6ed0000000000              // double -5869
	.xword	0x409d640000000000              // double 1881
	.xword	0xc0b1140000000000              // double -4372
	.xword	0xc0a19e0000000000              // double -2255
	.xword	0x4088700000000000              // double 782
	.xword	0x408d100000000000              // double 930
	.xword	0x408c880000000000              // double 913
	.xword	0x41010ec800000000              // double 139737
	.xword	0x0000000000000000              // double 0
	.xword	0x40d816c000000000              // double 24667
	.xword	0x40e8f66000000000              // double 51123
	.xword	0xc0b3ee0000000000              // double -5102
	.xword	0x40bd050000000000              // double 7429
	.xword	0xc0affe0000000000              // double -4095
	.xword	0xc09ee00000000000              // double -1976
	.xword	0xc0c2af0000000000              // double -9566
	.xword	0xc100db0800000000              // double -138081
	.xword	0x0000000000000000              // double 0
	.xword	0x40e22aa000000000              // double 37205
	.xword	0xc0e7f1e000000000              // double -49039
	.xword	0xc0e475a000000000              // double -41901
	.xword	0xc0e08a0000000000              // double -33872
	.xword	0xc0da674000000000              // double -27037
	.xword	0xc0c85d0000000000              // double -12474
	.xword	0x40d25b4000000000              // double 18797
	.xword	0x0000000000000000              // double 0
	.xword	0x40dbd30000000000              // double 28492
	.xword	0x410043a000000000              // double 133236
	.xword	0x40f1016000000000              // double 69654
	.xword	0x40e98c4000000000              // double 52322
	.xword	0xc0e8352000000000              // double -49577
	.xword	0xc0d9cf8000000000              // double -26430
	.xword	0xc0ac120000000000              // double -3593
	.xword	0x0000000000000000              // double 0
	.size	sa, 576

	.type	cl,@object                      // @cl
	.p2align	3, 0x0
cl:
	.xword	0x4035000000000000              // double 21
	.xword	0xc057c00000000000              // double -95
	.xword	0xc063a00000000000              // double -157
	.xword	0x4044800000000000              // double 41
	.xword	0xc014000000000000              // double -5
	.xword	0x4045000000000000              // double 42
	.xword	0x4037000000000000              // double 23
	.xword	0x403e000000000000              // double 30
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0xc064000000000000              // double -160
	.xword	0xc073900000000000              // double -313
	.xword	0xc06d600000000000              // double -235
	.xword	0x404e000000000000              // double 60
	.xword	0xc052800000000000              // double -74
	.xword	0xc053000000000000              // double -76
	.xword	0xc03b000000000000              // double -27
	.xword	0x4041000000000000              // double 34
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0xc074500000000000              // double -325
	.xword	0xc074200000000000              // double -322
	.xword	0xc053c00000000000              // double -79
	.xword	0x406d000000000000              // double 232
	.xword	0xc04a000000000000              // double -52
	.xword	0x4058400000000000              // double 97
	.xword	0x404b800000000000              // double 55
	.xword	0xc044800000000000              // double -41
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x40a1b80000000000              // double 2268
	.xword	0xc08e980000000000              // double -979
	.xword	0x4089100000000000              // double 802
	.xword	0x4082d00000000000              // double 602
	.xword	0xc084e00000000000              // double -668
	.xword	0xc040800000000000              // double -33
	.xword	0x4075900000000000              // double 345
	.xword	0x4069200000000000              // double 201
	.xword	0xc04b800000000000              // double -55
	.xword	0x0000000000000000              // double 0
	.xword	0x40bdba0000000000              // double 7610
	.xword	0xc0b3850000000000              // double -4997
	.xword	0xc0be090000000000              // double -7689
	.xword	0xc0b6d10000000000              // double -5841
	.xword	0xc0a4720000000000              // double -2617
	.xword	0x40916c0000000000              // double 1115
	.xword	0xc087600000000000              // double -748
	.xword	0xc082f80000000000              // double -607
	.xword	0x40b7ba0000000000              // double 6074
	.xword	0x4076200000000000              // double 354
	.xword	0xc0d21d4000000000              // double -18549
	.xword	0x40dd6b4000000000              // double 30125
	.xword	0x40d38b0000000000              // double 20012
	.xword	0xc086d00000000000              // double -730
	.xword	0x4089c00000000000              // double 824
	.xword	0x4037000000000000              // double 23
	.xword	0x4094240000000000              // double 1289
	.xword	0xc076000000000000              // double -352
	.xword	0xc0ccd78000000000              // double -14767
	.xword	0xc0a01c0000000000              // double -2062
	.xword	0xc100826800000000              // double -135245
	.xword	0xc0cc810000000000              // double -14594
	.xword	0x40b0650000000000              // double 4197
	.xword	0xc0af7c0000000000              // double -4030
	.xword	0xc0b5fe0000000000              // double -5630
	.xword	0xc0a6a40000000000              // double -2898
	.xword	0x40a3d80000000000              // double 2540
	.xword	0xc073200000000000              // double -306
	.xword	0x40a6f60000000000              // double 2939
	.xword	0x409f080000000000              // double 1986
	.xword	0x40f5f5c000000000              // double 89948
	.xword	0x40a06e0000000000              // double 2103
	.xword	0x40c1818000000000              // double 8963
	.xword	0x40a50e0000000000              // double 2695
	.xword	0x40acc40000000000              // double 3682
	.xword	0x4099c00000000000              // double 1648
	.xword	0x408b100000000000              // double 866
	.xword	0xc063400000000000              // double -154
	.xword	0xc09eac0000000000              // double -1963
	.xword	0xc071b00000000000              // double -283
	.size	cl, 640

	.type	sl,@object                      // @sl
	.p2align	3, 0x0
sl:
	.xword	0xc075600000000000              // double -342
	.xword	0x4061000000000000              // double 136
	.xword	0xc037000000000000              // double -23
	.xword	0x404f000000000000              // double 62
	.xword	0x4050800000000000              // double 66
	.xword	0xc04a000000000000              // double -52
	.xword	0xc040800000000000              // double -33
	.xword	0x4031000000000000              // double 17
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x4080600000000000              // double 524
	.xword	0xc062a00000000000              // double -149
	.xword	0xc041800000000000              // double -35
	.xword	0x405d400000000000              // double 117
	.xword	0x4062e00000000000              // double 151
	.xword	0x405e800000000000              // double 122
	.xword	0xc051c00000000000              // double -71
	.xword	0xc04f000000000000              // double -62
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0xc05a400000000000              // double -105
	.xword	0xc061200000000000              // double -137
	.xword	0x4070200000000000              // double 258
	.xword	0x4041800000000000              // double 35
	.xword	0xc05d000000000000              // double -116
	.xword	0xc056000000000000              // double -88
	.xword	0xc05c000000000000              // double -112
	.xword	0xc054000000000000              // double -80
	.xword	0x0000000000000000              // double 0
	.xword	0x0000000000000000              // double 0
	.xword	0x408ab00000000000              // double 854
	.xword	0xc069a00000000000              // double -205
	.xword	0xc08d400000000000              // double -936
	.xword	0xc06e000000000000              // double -240
	.xword	0x4061800000000000              // double 140
	.xword	0xc075500000000000              // double -341
	.xword	0xc058400000000000              // double -97
	.xword	0xc06d000000000000              // double -232
	.xword	0x4080c00000000000              // double 536
	.xword	0x0000000000000000              // double 0
	.xword	0xc0ebd28000000000              // double -56980
	.xword	0x40bf500000000000              // double 8016
	.xword	0x408fa00000000000              // double 1012
	.xword	0x4096a00000000000              // double 1448
	.xword	0xc0a7a00000000000              // double -3024
	.xword	0xc0acfc0000000000              // double -3710
	.xword	0x4073e00000000000              // double 318
	.xword	0x407f700000000000              // double 503
	.xword	0x40ad6e0000000000              // double 3767
	.xword	0x4082080000000000              // double 577
	.xword	0x4100eb7000000000              // double 138606
	.xword	0xc0ca530000000000              // double -13478
	.xword	0xc0b3640000000000              // double -4964
	.xword	0x4096840000000000              // double 1441
	.xword	0xc0949c0000000000              // double -1319
	.xword	0xc097280000000000              // double -1482
	.xword	0x407ab00000000000              // double 427
	.xword	0x4093500000000000              // double 1236
	.xword	0xc0c1e78000000000              // double -9167
	.xword	0xc09df80000000000              // double -1918
	.xword	0x40f1642000000000              // double 71234
	.xword	0xc0e4138000000000              // double -41116
	.xword	0x40b4d60000000000              // double 5334
	.xword	0xc0b3470000000000              // double -4935
	.xword	0xc09ce00000000000              // double -1848
	.xword	0x4050800000000000              // double 66
	.xword	0x407b200000000000              // double 434
	.xword	0xc09b500000000000              // double -1748
	.xword	0x40ad880000000000              // double 3780
	.xword	0xc085e80000000000              // double -701
	.xword	0xc0e743a000000000              // double -47645
	.xword	0x40c6bf8000000000              // double 11647
	.xword	0x40a0ec0000000000              // double 2166
	.xword	0x40a8f40000000000              // double 3194
	.xword	0x4085380000000000              // double 679
	.xword	0x0000000000000000              // double 0
	.xword	0xc06e800000000000              // double -244
	.xword	0xc07a300000000000              // double -419
	.xword	0xc0a3c60000000000              // double -2531
	.xword	0x4048000000000000              // double 48
	.size	sl, 640

	.type	amas,@object                    // @amas
	.p2align	3, 0x0
amas:
	.xword	0x4156fa6c00000000              // double 6023600
	.xword	0x4118ef2e00000000              // double 408523.5
	.xword	0x4114131200000000              // double 328900.5
	.xword	0x4147a42b00000000              // double 3098710
	.xword	0x40905d6b851eb852              // double 1047.355
	.xword	0x40ab550000000000              // double 3498.5
	.xword	0x40d6554000000000              // double 22869
	.xword	0x40d2dc8000000000              // double 19314
	.size	amas, 64

	.type	.L.str.1,@object                // @.str.1
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str.1:
	.asciz	"%f %f %f\n"
	.size	.L.str.1, 10

	.ident	"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"
	.section	".note.GNU-stack","",@progbits
