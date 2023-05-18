@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

        vadd.f64  d16, d17, d16
        vadd.f32  s0, s1, s0
@ CHECK: vadd.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x71,0xee]
@ CHECK: vadd.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x30,0xee]

        vsub.f64  d16, d17, d16
        vsub.f32  s0, s1, s0
@ CHECK: vsub.f64 d16, d17, d16      @ encoding: [0xe0,0x0b,0x71,0xee]
@ CHECK: vsub.f32 s0, s1, s0         @ encoding: [0xc0,0x0a,0x30,0xee]

        vdiv.f64  d16, d17, d16
        vdiv.f32  s0, s1, s0
        vdiv.f32 s5, s7
        vdiv.f64 d5, d7

@ CHECK: vdiv.f64 d16, d17, d16         @ encoding: [0xa0,0x0b,0xc1,0xee]
@ CHECK: vdiv.f32 s0, s1, s0            @ encoding: [0x80,0x0a,0x80,0xee]
@ CHECK: vdiv.f32	s5, s5, s7      @ encoding: [0xa3,0x2a,0xc2,0xee]
@ CHECK: vdiv.f64	d5, d5, d7      @ encoding: [0x07,0x5b,0x85,0xee]


        vmul.f64  d16, d17, d16
	vmul.f64  d20, d17
        vmul.f32  s0, s1, s0
	vmul.f32  s11, s21


@ CHECK: vmul.f64 d16, d17, d16      @ encoding: [0xa0,0x0b,0x61,0xee]
@ CHECK: vmul.f64 d20, d20, d17      @ encoding: [0xa1,0x4b,0x64,0xee]
@ CHECK: vmul.f32 s0, s1, s0         @ encoding: [0x80,0x0a,0x20,0xee]
@ CHECK: vmul.f32 s11, s11, s21      @ encoding: [0xaa,0x5a,0x65,0xee]

        vnmul.f64       d16, d17, d16
        vnmul.f32       s0, s1, s0
        vnmul.f64       d0, d1
        vnmul.f32       s0, s1

@ CHECK: vnmul.f64 d16, d17, d16     @ encoding: [0xe0,0x0b,0x61,0xee]
@ CHECK: vnmul.f32 s0, s1, s0        @ encoding: [0xc0,0x0a,0x20,0xee]
@ CHECK: vnmul.f64 d0, d0, d1        @ encoding: [0x41,0x0b,0x20,0xee]
@ CHECK: vnmul.f32 s0, s0, s1        @ encoding: [0x60,0x0a,0x20,0xee]

        vcmp.f64       d17, d16
        vcmp.f32       s1, s0

@ CHECK: vcmp.f64  d17, d16        @ encoding: [0x60,0x1b,0xf4,0xee]
@ CHECK: vcmp.f32  s1, s0          @ encoding: [0x40,0x0a,0xf4,0xee]

        vcmp.f64       d17, #0.0
        vcmp.f32       s1, #0.0

@ CHECK: vcmp.f64  d17, #0         @ encoding: [0x40,0x1b,0xf5,0xee]
@ CHECK: vcmp.f32  s1, #0          @ encoding: [0x40,0x0a,0xf5,0xee]

        vcmpe.f64       d17, d16
        vcmpe.f32       s1, s0

@ CHECK: vcmpe.f64 d17, d16          @ encoding: [0xe0,0x1b,0xf4,0xee]
@ CHECK: vcmpe.f32 s1, s0            @ encoding: [0xc0,0x0a,0xf4,0xee]

        vcmpe.f64       d16, #0
        vcmpe.f32       s0, #0

@ CHECK: vcmpe.f64 d16, #0           @ encoding: [0xc0,0x0b,0xf5,0xee]
@ CHECK: vcmpe.f32 s0, #0            @ encoding: [0xc0,0x0a,0xb5,0xee]

        vabs.f64        d16, d16
        vabs.f32        s0, s0

@ CHECK: vabs.f64 d16, d16           @ encoding: [0xe0,0x0b,0xf0,0xee]
@ CHECK: vabs.f32 s0, s0             @ encoding: [0xc0,0x0a,0xb0,0xee]

        vcvt.f32.f64    s0, d16
        vcvt.f64.f32    d16, s0

@ CHECK: vcvt.f32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xb7,0xee]
@ CHECK: vcvt.f64.f32 d16, s0        @ encoding: [0xc0,0x0a,0xf7,0xee]

        vneg.f64        d16, d16
        vneg.f32        s0, s0

@ CHECK: vneg.f64 d16, d16           @ encoding: [0x60,0x0b,0xf1,0xee]
@ CHECK: vneg.f32 s0, s0             @ encoding: [0x40,0x0a,0xb1,0xee]

        vsqrt.f64       d16, d16
        vsqrt.f32       s0, s0

@ CHECK: vsqrt.f64 d16, d16          @ encoding: [0xe0,0x0b,0xf1,0xee]
@ CHECK: vsqrt.f32 s0, s0            @ encoding: [0xc0,0x0a,0xb1,0xee]

        vcvt.f64.s32    d16, s0
        vcvt.f32.s32    s0, s0
        vcvt.f64.u32    d16, s0
        vcvt.f32.u32    s0, s0
        vcvt.s32.f64    s0, d16
        vcvt.s32.f32    s0, s0
        vcvt.u32.f64    s0, d16
        vcvt.u32.f32    s0, s0

@ CHECK: vcvt.f64.s32 d16, s0        @ encoding: [0xc0,0x0b,0xf8,0xee]
@ CHECK: vcvt.f32.s32 s0, s0         @ encoding: [0xc0,0x0a,0xb8,0xee]
@ CHECK: vcvt.f64.u32 d16, s0        @ encoding: [0x40,0x0b,0xf8,0xee]
@ CHECK: vcvt.f32.u32 s0, s0         @ encoding: [0x40,0x0a,0xb8,0xee]
@ CHECK: vcvt.s32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbd,0xee]
@ CHECK: vcvt.s32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbd,0xee]
@ CHECK: vcvt.u32.f64 s0, d16        @ encoding: [0xe0,0x0b,0xbc,0xee]
@ CHECK: vcvt.u32.f32 s0, s0         @ encoding: [0xc0,0x0a,0xbc,0xee]


        vmla.f64        d16, d18, d17
        vmla.f32        s1, s2, s0

@ CHECK: vmla.f64 d16, d18, d17      @ encoding: [0xa1,0x0b,0x42,0xee]
@ CHECK: vmla.f32 s1, s2, s0         @ encoding: [0x00,0x0a,0x41,0xee]

        vmls.f64        d16, d18, d17
        vmls.f32        s1, s2, s0

@ CHECK: vmls.f64 d16, d18, d17      @ encoding: [0xe1,0x0b,0x42,0xee]
@ CHECK: vmls.f32 s1, s2, s0         @ encoding: [0x40,0x0a,0x41,0xee]

        vnmla.f64       d16, d18, d17
        vnmla.f32       s1, s2, s0

@ CHECK: vnmla.f64 d16, d18, d17     @ encoding: [0xe1,0x0b,0x52,0xee]
@ CHECK: vnmla.f32 s1, s2, s0        @ encoding: [0x40,0x0a,0x51,0xee]

        vnmls.f64       d16, d18, d17
        vnmls.f32       s1, s2, s0

@ CHECK: vnmls.f64 d16, d18, d17     @ encoding: [0xa1,0x0b,0x52,0xee]
@ CHECK: vnmls.f32 s1, s2, s0        @ encoding: [0x00,0x0a,0x51,0xee]

        vmrs    APSR_nzcv, fpscr
        vmrs    apsr_nzcv, fpscr
        fmstat
        vmrs    r2, fpsid
        vmrs    r3, FPSID
        vmrs    r4, mvfr0
        vmrs    r5, MVFR1

@ CHECK: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
@ CHECK: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
@ CHECK: vmrs APSR_nzcv, fpscr       @ encoding: [0x10,0xfa,0xf1,0xee]
@ CHECK: vmrs r2, fpsid              @ encoding: [0x10,0x2a,0xf0,0xee]
@ CHECK: vmrs r3, fpsid              @ encoding: [0x10,0x3a,0xf0,0xee]
@ CHECK: vmrs r4, mvfr0              @ encoding: [0x10,0x4a,0xf7,0xee]
@ CHECK: vmrs r5, mvfr1              @ encoding: [0x10,0x5a,0xf6,0xee]

@ CHECK: vnegne.f64 d16, d16         @ encoding: [0x60,0x0b,0xf1,0x1e]
        vnegne.f64      d16, d16

@ CHECK: vmovne s0, r0               @ encoding: [0x10,0x0a,0x00,0x1e]
@ CHECK: vmoveq s0, r1               @ encoding: [0x10,0x1a,0x00,0x0e]
        vmovne  s0, r0
        vmoveq  s0, r1

        vmov.f32 r1, s2
        vmov.f32 s4, r3
        vmov.f64 r1, r5, d2
        vmov.f64 d4, r3, r9

@ CHECK: vmov	r1, s2                  @ encoding: [0x10,0x1a,0x11,0xee]
@ CHECK: vmov	s4, r3                  @ encoding: [0x10,0x3a,0x02,0xee]
@ CHECK: vmov	r1, r5, d2              @ encoding: [0x12,0x1b,0x55,0xec]
@ CHECK: vmov	d4, r3, r9              @ encoding: [0x14,0x3b,0x49,0xec]

@ CHECK: vmrs r0, fpscr              @ encoding: [0x10,0x0a,0xf1,0xee]
        vmrs    r0, fpscr
@ CHECK: vmrs  r0, fpexc             @ encoding: [0x10,0x0a,0xf8,0xee]
        vmrs  r0, fpexc
@ CHECK: vmrs  r0, fpsid             @ encoding: [0x10,0x0a,0xf0,0xee]
        vmrs  r0, fpsid
@ CHECK: vmrs	r1, fpinst           @ encoding: [0x10,0x1a,0xf9,0xee]
        vmrs r1, fpinst
@ CHECK: vmrs	r8, fpinst2          @ encoding: [0x10,0x8a,0xfa,0xee]
        vmrs r8, fpinst2

@ CHECK: vmsr fpscr, r0              @ encoding: [0x10,0x0a,0xe1,0xee]
        vmsr    fpscr, r0
@ CHECK: vmsr  fpexc, r0             @ encoding: [0x10,0x0a,0xe8,0xee]
        vmsr  fpexc, r0
@ CHECK: vmsr  fpsid, r0             @ encoding: [0x10,0x0a,0xe0,0xee]
        vmsr  fpsid, r0
@ CHECK: vmsr	fpinst, r3           @ encoding: [0x10,0x3a,0xe9,0xee]
        vmsr fpinst, r3	
@ CHECK: vmsr	fpinst2, r4          @ encoding: [0x10,0x4a,0xea,0xee]
        vmsr fpinst2, r4

        vmov.f64        d16, #3.000000e+00
        vmov.f32        s0, #3.000000e+00
        vmov.f64        d16, #-3.000000e+00
        vmov.f32        s0, #-3.000000e+00

@ CHECK: vmov.f64 d16, #3.000000e+00 @ encoding: [0x08,0x0b,0xf0,0xee]
@ CHECK: vmov.f32 s0, #3.000000e+00  @ encoding: [0x08,0x0a,0xb0,0xee]
@ CHECK: vmov.f64 d16, #-3.000000e+00 @ encoding: [0x08,0x0b,0xf8,0xee]
@ CHECK: vmov.f32 s0, #-3.000000e+00  @ encoding: [0x08,0x0a,0xb8,0xee]

@ CHECK: vmov s0, r0                 @ encoding: [0x10,0x0a,0x00,0xee]
@ CHECK: vmov s1, r1                 @ encoding: [0x90,0x1a,0x00,0xee]
@ CHECK: vmov s2, r2                 @ encoding: [0x10,0x2a,0x01,0xee]
@ CHECK: vmov s3, r3                 @ encoding: [0x90,0x3a,0x01,0xee]
        vmov    s0, r0
        vmov    s1, r1
        vmov    s2, r2
        vmov    s3, r3

@ CHECK: vmov r0, s0                 @ encoding: [0x10,0x0a,0x10,0xee]
@ CHECK: vmov r1, s1                 @ encoding: [0x90,0x1a,0x10,0xee]
@ CHECK: vmov r2, s2                 @ encoding: [0x10,0x2a,0x11,0xee]
@ CHECK: vmov r3, s3                 @ encoding: [0x90,0x3a,0x11,0xee]
        vmov    r0, s0
        vmov    r1, s1
        vmov    r2, s2
        vmov    r3, s3

@ CHECK: vmov r0, r1, d16            @ encoding: [0x30,0x0b,0x51,0xec]
        vmov    r0, r1, d16

@ Between two single precision registers and two core registers
        vmov s3, s4, r1, r2
        vmov s2, s3, r1, r2
        vmov r1, r2, s3, s4
        vmov r1, r2, s2, s3
@ CHECK: vmov s3, s4, r1, r2      @ encoding: [0x31,0x1a,0x42,0xec]
@ CHECK: vmov s2, s3, r1, r2      @ encoding: [0x11,0x1a,0x42,0xec]
@ CHECK: vmov r1, r2, s3, s4      @ encoding: [0x31,0x1a,0x52,0xec]
@ CHECK: vmov r1, r2, s2, s3      @ encoding: [0x11,0x1a,0x52,0xec]

@ Between one double precision register and two core registers
        vmov d15, r1, r2 
        vmov d16, r1, r2
        vmov r1, r2, d15
        vmov r1, r2, d16
@ CHECK: vmov d15, r1, r2         @ encoding: [0x1f,0x1b,0x42,0xec]
@ CHECK: vmov d16, r1, r2         @ encoding: [0x30,0x1b,0x42,0xec]
@ CHECK: vmov r1, r2, d15         @ encoding: [0x1f,0x1b,0x52,0xec]
@ CHECK: vmov r1, r2, d16         @ encoding: [0x30,0x1b,0x52,0xec]


@ CHECK: vldr d17, [r0]           @ encoding: [0x00,0x1b,0xd0,0xed]
@ CHECK: vldr s0, [lr]            @ encoding: [0x00,0x0a,0x9e,0xed]
@ CHECK: vldr d0, [lr]            @ encoding: [0x00,0x0b,0x9e,0xed]

        vldr.64	d17, [r0]
	vldr.i32 s0, [lr]
	vldr.d d0, [lr]

@ CHECK: vldr d1, [r2, #32]       @ encoding: [0x08,0x1b,0x92,0xed]
@ CHECK: vldr d1, [r2, #-32]      @ encoding: [0x08,0x1b,0x12,0xed]
        vldr.64	d1, [r2, #32]
        vldr.f64	d1, [r2, #-32]

@ CHECK: vldr d2, [r3]            @ encoding: [0x00,0x2b,0x93,0xed]
        vldr.64 d2, [r3]

@ CHECK: vldr d3, [pc]            @ encoding: [0x00,0x3b,0x9f,0xed]
@ CHECK: vldr d3, [pc]            @ encoding: [0x00,0x3b,0x9f,0xed]
@ CHECK: vldr d3, [pc, #-0]            @ encoding: [0x00,0x3b,0x1f,0xed]
        vldr.64 d3, [pc]
        vldr.64 d3, [pc,#0]
        vldr.64 d3, [pc,#-0]

@ CHECK: vldr s13, [r0]           @ encoding: [0x00,0x6a,0xd0,0xed]
        vldr.32	s13, [r0]

@ CHECK: vldr s1, [r2, #32]       @ encoding: [0x08,0x0a,0xd2,0xed]
@ CHECK: vldr s1, [r2, #-32]      @ encoding: [0x08,0x0a,0x52,0xed]
        vldr.32	s1, [r2, #32]
        vldr.32	s1, [r2, #-32]

@ CHECK: vldr s2, [r3]            @ encoding: [0x00,0x1a,0x93,0xed]
        vldr.32 s2, [r3]

@ CHECK: vldr s5, [pc]            @ encoding: [0x00,0x2a,0xdf,0xed]
@ CHECK: vldr s5, [pc]            @ encoding: [0x00,0x2a,0xdf,0xed]
@ CHECK: vldr s5, [pc, #-0]            @ encoding: [0x00,0x2a,0x5f,0xed]
        vldr.32 s5, [pc]
        vldr.32 s5, [pc,#0]
        vldr.32 s5, [pc,#-0]

@ CHECK: vstr d4, [r1]            @ encoding: [0x00,0x4b,0x81,0xed]
@ CHECK: vstr d4, [r1, #24]       @ encoding: [0x06,0x4b,0x81,0xed]
@ CHECK: vstr d4, [r1, #-24]      @ encoding: [0x06,0x4b,0x01,0xed]
@ CHECK: vstr s0, [lr]            @ encoding: [0x00,0x0a,0x8e,0xed]
@ CHECK: vstr d0, [lr]            @ encoding: [0x00,0x0b,0x8e,0xed]

        vstr.64 d4, [r1]
        vstr.64 d4, [r1, #24]
        vstr.64 d4, [r1, #-24]
	vstr s0, [lr]
	vstr d0, [lr]

@ CHECK: vstr s4, [r1]            @ encoding: [0x00,0x2a,0x81,0xed]
@ CHECK: vstr s4, [r1, #24]       @ encoding: [0x06,0x2a,0x81,0xed]
@ CHECK: vstr s4, [r1, #-24]      @ encoding: [0x06,0x2a,0x01,0xed]
        vstr.32 s4, [r1]
        vstr.32 s4, [r1, #24]
        vstr.32 s4, [r1, #-24]

@ CHECK: vldmia r1, {d2, d3, d4, d5, d6, d7} @ encoding: [0x0c,0x2b,0x91,0xec]
@ CHECK: vldmia r1, {s2, s3, s4, s5, s6, s7} @ encoding: [0x06,0x1a,0x91,0xec]
        vldmia  r1, {d2,d3-d6,d7}
        vldmia  r1, {s2,s3-s6,s7}

@ CHECK: vstmia r1, {d2, d3, d4, d5, d6, d7} @ encoding: [0x0c,0x2b,0x81,0xec]
@ CHECK: vstmia	r1, {s2, s3, s4, s5, s6, s7} @ encoding: [0x06,0x1a,0x81,0xec]
@ CHECK: vpush	{d8, d9, d10, d11, d12, d13, d14, d15} @ encoding: [0x10,0x8b,0x2d,0xed]
        vstmia  r1, {d2,d3-d6,d7}
        vstmia  r1, {s2,s3-s6,s7}
        vstmdb sp!, {q4-q7}

        fldmiax r5!, {d0-d2}
        fldmiaxeq r0, {d4,d5}
        fldmdbxne r5!, {d4,d5,d6}
@ CHECK: fldmiax r5!, {d0, d1, d2}      @ encoding: [0x07,0x0b,0xb5,0xec]
@ CHECK: fldmiaxeq r0, {d4, d5}         @ encoding: [0x05,0x4b,0x90,0x0c]
@ CHECK: fldmdbxne r5!, {d4, d5, d6}    @ encoding: [0x07,0x4b,0x35,0x1d]

        fstmiax r5!, {d0-d7}
        fstmiaxeq r4, {d8,d9}
        fstmdbxne r7!, {d2-d4}
@ CHECK: fstmiax r5!, {d0, d1, d2, d3, d4, d5, d6, d7} @ encoding: [0x11,0x0b,0xa5,0xec]
@ CHECK: fstmiaxeq r4, {d8, d9}         @ encoding: [0x05,0x8b,0x84,0x0c]
@ CHECK: fstmdbxne r7!, {d2, d3, d4}    @ encoding: [0x07,0x2b,0x27,0x1d]

@ CHECK: vcvtr.s32.f64  s0, d0 @ encoding: [0x40,0x0b,0xbd,0xee]
@ CHECK: vcvtr.s32.f32  s0, s1 @ encoding: [0x60,0x0a,0xbd,0xee]
@ CHECK: vcvtr.u32.f64  s0, d0 @ encoding: [0x40,0x0b,0xbc,0xee]
@ CHECK: vcvtr.u32.f32  s0, s1 @ encoding: [0x60,0x0a,0xbc,0xee]
        vcvtr.s32.f64  s0, d0
        vcvtr.s32.f32  s0, s1
        vcvtr.u32.f64  s0, d0
        vcvtr.u32.f32  s0, s1

@ CHECK: vmovne	s25, s26, r2, r5
        vmovne	s25, s26, r2, r5        @ encoding: [0x39,0x2a,0x45,0x1c]

@ VMOV w/ optional data type suffix.
	vmov.32 s1, r8
        vmov.s16 s2, r4
        vmov.16 s3, r6
        vmov.u32 s4, r1
        vmov.p8 s5, r2
        vmov.8 s6, r3

        vmov.32 r1, s8
        vmov.s16 r2, s4
        vmov.16 r3, s6
        vmov.u32 r4, s1
        vmov.p8 r5, s2
        vmov.8 r6, s3

@ CHECK: vmov	s1, r8                  @ encoding: [0x90,0x8a,0x00,0xee]
@ CHECK: vmov	s2, r4                  @ encoding: [0x10,0x4a,0x01,0xee]
@ CHECK: vmov	s3, r6                  @ encoding: [0x90,0x6a,0x01,0xee]
@ CHECK: vmov	s4, r1                  @ encoding: [0x10,0x1a,0x02,0xee]
@ CHECK: vmov	s5, r2                  @ encoding: [0x90,0x2a,0x02,0xee]
@ CHECK: vmov	s6, r3                  @ encoding: [0x10,0x3a,0x03,0xee]
@ CHECK: vmov	r1, s8                  @ encoding: [0x10,0x1a,0x14,0xee]
@ CHECK: vmov	r2, s4                  @ encoding: [0x10,0x2a,0x12,0xee]
@ CHECK: vmov	r3, s6                  @ encoding: [0x10,0x3a,0x13,0xee]
@ CHECK: vmov	r4, s1                  @ encoding: [0x90,0x4a,0x10,0xee]
@ CHECK: vmov	r5, s2                  @ encoding: [0x10,0x5a,0x11,0xee]
@ CHECK: vmov	r6, s3                  @ encoding: [0x90,0x6a,0x11,0xee]


@ VCVT (between floating-point and fixed-point)
        vcvt.f32.u32 s0, s0, #20
        vcvt.f64.s32 d0, d0, #32
        vcvt.f32.u16 s0, s0, #1
        vcvt.f64.s16 d0, d0, #16
        vcvt.f32.s32 s1, s1, #20
        vcvt.f64.u32 d20, d20, #32
        vcvt.f32.s16 s17, s17, #1
        vcvt.f64.u16 d23, d23, #16
        vcvt.u32.f32 s12, s12, #20 
        vcvt.s32.f64 d2, d2, #32
        vcvt.u16.f32 s28, s28, #1
        vcvt.s16.f64 d15, d15, #16
        vcvt.s32.f32 s1, s1, #20
        vcvt.u32.f64 d20, d20, #32
        vcvt.s16.f32 s17, s17, #1
        vcvt.u16.f64 d23, d23, #16

@ CHECK: vcvt.f32.u32	s0, s0, #20     @ encoding: [0xc6,0x0a,0xbb,0xee]
@ CHECK: vcvt.f64.s32	d0, d0, #32     @ encoding: [0xc0,0x0b,0xba,0xee]
@ CHECK: vcvt.f32.u16	s0, s0, #1      @ encoding: [0x67,0x0a,0xbb,0xee]
@ CHECK: vcvt.f64.s16	d0, d0, #16     @ encoding: [0x40,0x0b,0xba,0xee]
@ CHECK: vcvt.f32.s32	s1, s1, #20     @ encoding: [0xc6,0x0a,0xfa,0xee]
@ CHECK: vcvt.f64.u32	d20, d20, #32   @ encoding: [0xc0,0x4b,0xfb,0xee]
@ CHECK: vcvt.f32.s16	s17, s17, #1    @ encoding: [0x67,0x8a,0xfa,0xee]
@ CHECK: vcvt.f64.u16	d23, d23, #16   @ encoding: [0x40,0x7b,0xfb,0xee]

@ CHECK: vcvt.u32.f32	s12, s12, #20   @ encoding: [0xc6,0x6a,0xbf,0xee]
@ CHECK: vcvt.s32.f64	d2, d2, #32     @ encoding: [0xc0,0x2b,0xbe,0xee]
@ CHECK: vcvt.u16.f32	s28, s28, #1    @ encoding: [0x67,0xea,0xbf,0xee]
@ CHECK: vcvt.s16.f64	d15, d15, #16   @ encoding: [0x40,0xfb,0xbe,0xee]
@ CHECK: vcvt.s32.f32	s1, s1, #20     @ encoding: [0xc6,0x0a,0xfe,0xee]
@ CHECK: vcvt.u32.f64	d20, d20, #32   @ encoding: [0xc0,0x4b,0xff,0xee]
@ CHECK: vcvt.s16.f32	s17, s17, #1    @ encoding: [0x67,0x8a,0xfe,0xee]
@ CHECK: vcvt.u16.f64	d23, d23, #16   @ encoding: [0x40,0x7b,0xff,0xee]


@ Use NEON to load some f32 immediates that don't fit the f8 representation.
        vmov.f32 d4, #0.0
        vmov.f32 d4, #32.0

@ CHECK: vmov.i32	d4, #0x0        @ encoding: [0x10,0x40,0x80,0xf2]
@ CHECK: vmov.i32	d4, #0x42000000 @ encoding: [0x12,0x46,0x84,0xf2]

@ Test encoding of floating point constants for vmov functions
@ vfp3
         vmov.f32 s5, #1.0
         vmov.f32 s5, #0.125
         vmov.f32 s5, #-1.875
         vmov.f32 s5, #-0.59375

         vmov.f64 d6, #1.0
         vmov.f64 d6, #0.125
         vmov.f64 d6, #-1.875
         vmov.f64 d6, #-0.59375

@ neon
         vmov.f32 d7, #1.0
         vmov.f32 d7, #0.125
         vmov.f32 d7, #-1.875
         vmov.f32 d7, #-0.59375

         vmov.f32 q8, #1.0
         vmov.f32 q8, #0.125
         vmov.f32 q8, #-1.875
         vmov.f32 q8, #-0.59375

@ CHECK: vmov.f32        s5, #1.000000e+00 @ encoding: [0x00,0x2a,0xf7,0xee]
@ CHECK: vmov.f32        s5, #1.250000e-01 @ encoding: [0x00,0x2a,0xf4,0xee]
@ CHECK: vmov.f32        s5, #-1.875000e+00 @ encoding: [0x0e,0x2a,0xff,0xee]
@ CHECK: vmov.f32        s5, #-5.937500e-01 @ encoding: [0x03,0x2a,0xfe,0xee]

@ CHECK: vmov.f64        d6, #1.000000e+00 @ encoding: [0x00,0x6b,0xb7,0xee]
@ CHECK: vmov.f64        d6, #1.250000e-01 @ encoding: [0x00,0x6b,0xb4,0xee]
@ CHECK: vmov.f64        d6, #-1.875000e+00 @ encoding: [0x0e,0x6b,0xbf,0xee]
@ CHECK: vmov.f64        d6, #-5.937500e-01 @ encoding: [0x03,0x6b,0xbe,0xee]

@ CHECK: vmov.f32        d7, #1.000000e+00 @ encoding: [0x10,0x7f,0x87,0xf2]
@ CHECK: vmov.f32        d7, #1.250000e-01 @ encoding: [0x10,0x7f,0x84,0xf2]
@ CHECK: vmov.f32        d7, #-1.875000e+00 @ encoding: [0x1e,0x7f,0x87,0xf3]
@ CHECK: vmov.f32        d7, #-5.937500e-01 @ encoding: [0x13,0x7f,0x86,0xf3]

@ CHECK: vmov.f32        q8, #1.000000e+00 @ encoding: [0x50,0x0f,0xc7,0xf2]
@ CHECK: vmov.f32        q8, #1.250000e-01 @ encoding: [0x50,0x0f,0xc4,0xf2]
@ CHECK: vmov.f32        q8, #-1.875000e+00 @ encoding: [0x5e,0x0f,0xc7,0xf3]
@ CHECK: vmov.f32        q8, #-5.937500e-01 @ encoding: [0x53,0x0f,0xc6,0xf3]
