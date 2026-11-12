@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16
	vpadd.i8	d1, d5, d11
	vpadd.i16	d13, d2, d12
	vpadd.i32	d14, d1, d13
	vpadd.f32	d19, d16, d14

@ CHECK: vpadd.i8	d1, d5, d11     @ encoding: [0x05,0xef,0x1b,0x1b]
@ CHECK: vpadd.i16	d13, d2, d12    @ encoding: [0x12,0xef,0x1c,0xdb]
@ CHECK: vpadd.i32	d14, d1, d13    @ encoding: [0x21,0xef,0x1d,0xeb]
@ CHECK: vpadd.f32	d19, d16, d14   @ encoding: [0x40,0xff,0x8e,0x3d]


	vpaddl.s8	d7, d10
	vpaddl.s16	d8, d11
	vpaddl.s32	d9, d12
	vpaddl.u8	d0, d13
	vpaddl.u16	d5, d14
	vpaddl.u32	d6, d15
	vpaddl.s8	q4, q7
	vpaddl.s16	q5, q6
	vpaddl.s32	q6, q5
	vpaddl.u8	q7, q4
	vpaddl.u16	q8, q3
	vpaddl.u32	q9, q2

@ CHECK: vpaddl.s8	d7, d10         @ encoding: [0xb0,0xff,0x0a,0x72]
@ CHECK: vpaddl.s16	d8, d11         @ encoding: [0xb4,0xff,0x0b,0x82]
@ CHECK: vpaddl.s32	d9, d12         @ encoding: [0xb8,0xff,0x0c,0x92]
@ CHECK: vpaddl.u8	d0, d13         @ encoding: [0xb0,0xff,0x8d,0x02]
@ CHECK: vpaddl.u16	d5, d14         @ encoding: [0xb4,0xff,0x8e,0x52]
@ CHECK: vpaddl.u32	d6, d15         @ encoding: [0xb8,0xff,0x8f,0x62]
@ CHECK: vpaddl.s8	q4, q7          @ encoding: [0xb0,0xff,0x4e,0x82]
@ CHECK: vpaddl.s16	q5, q6          @ encoding: [0xb4,0xff,0x4c,0xa2]
@ CHECK: vpaddl.s32	q6, q5          @ encoding: [0xb8,0xff,0x4a,0xc2]
@ CHECK: vpaddl.u8	q7, q4          @ encoding: [0xb0,0xff,0xc8,0xe2]
@ CHECK: vpaddl.u16	q8, q3          @ encoding: [0xf4,0xff,0xc6,0x02]
@ CHECK: vpaddl.u32	q9, q2          @ encoding: [0xf8,0xff,0xc4,0x22]


	vpadal.s8	d16, d4
	vpadal.s16	d20, d9
	vpadal.s32	d18, d1
	vpadal.u8	d14, d25
	vpadal.u16	d12, d6
	vpadal.u32	d11, d7
	vpadal.s8	q4, q10
	vpadal.s16	q5, q11
	vpadal.s32	q6, q12
	vpadal.u8	q7, q13
	vpadal.u16	q8, q14
	vpadal.u32	q9, q15

@ CHECK: vpadal.s8	d16, d4         @ encoding: [0xf0,0xff,0x04,0x06]
@ CHECK: vpadal.s16	d20, d9         @ encoding: [0xf4,0xff,0x09,0x46]
@ CHECK: vpadal.s32	d18, d1         @ encoding: [0xf8,0xff,0x01,0x26]
@ CHECK: vpadal.u8	d14, d25        @ encoding: [0xb0,0xff,0xa9,0xe6]
@ CHECK: vpadal.u16	d12, d6         @ encoding: [0xb4,0xff,0x86,0xc6]
@ CHECK: vpadal.u32	d11, d7         @ encoding: [0xb8,0xff,0x87,0xb6]
@ CHECK: vpadal.s8	q4, q10         @ encoding: [0xb0,0xff,0x64,0x86]
@ CHECK: vpadal.s16	q5, q11         @ encoding: [0xb4,0xff,0x66,0xa6]
@ CHECK: vpadal.s32	q6, q12         @ encoding: [0xb8,0xff,0x68,0xc6]
@ CHECK: vpadal.u8	q7, q13         @ encoding: [0xb0,0xff,0xea,0xe6]
@ CHECK: vpadal.u16	q8, q14         @ encoding: [0xf4,0xff,0xec,0x06]
@ CHECK: vpadal.u32	q9, q15         @ encoding: [0xf8,0xff,0xee,0x26]


	vpmin.s8	d16, d29, d10
	vpmin.s16	d17, d28, d11
	vpmin.s32	d18, d27, d12
	vpmin.u8	d19, d26, d13
	vpmin.u16	d20, d25, d14
	vpmin.u32	d21, d24, d15
	vpmin.f32	d22, d23, d16

@ CHECK: vpmin.s8	d16, d29, d10   @ encoding: [0x4d,0xef,0x9a,0x0a]
@ CHECK: vpmin.s16	d17, d28, d11   @ encoding: [0x5c,0xef,0x9b,0x1a]
@ CHECK: vpmin.s32	d18, d27, d12   @ encoding: [0x6b,0xef,0x9c,0x2a]
@ CHECK: vpmin.u8	d19, d26, d13   @ encoding: [0x4a,0xff,0x9d,0x3a]
@ CHECK: vpmin.u16	d20, d25, d14   @ encoding: [0x59,0xff,0x9e,0x4a]
@ CHECK: vpmin.u32	d21, d24, d15   @ encoding: [0x68,0xff,0x9f,0x5a]
@ CHECK: vpmin.f32	d22, d23, d16   @ encoding: [0x67,0xff,0xa0,0x6f]


	vpmax.s8	d3, d20, d17
	vpmax.s16	d4, d21, d16
	vpmax.s32	d5, d22, d15
	vpmax.u8	d6, d23, d14
	vpmax.u16	d7, d24, d13
	vpmax.u32	d8, d25, d12
	vpmax.f32	d9, d26, d11

@ CHECK: vpmax.s8	d3, d20, d17    @ encoding: [0x04,0xef,0xa1,0x3a]
@ CHECK: vpmax.s16	d4, d21, d16    @ encoding: [0x15,0xef,0xa0,0x4a]
@ CHECK: vpmax.s32	d5, d22, d15    @ encoding: [0x26,0xef,0x8f,0x5a]
@ CHECK: vpmax.u8	d6, d23, d14    @ encoding: [0x07,0xff,0x8e,0x6a]
@ CHECK: vpmax.u16	d7, d24, d13    @ encoding: [0x18,0xff,0x8d,0x7a]
@ CHECK: vpmax.u32	d8, d25, d12    @ encoding: [0x29,0xff,0x8c,0x8a]
@ CHECK: vpmax.f32	d9, d26, d11    @ encoding: [0x0a,0xff,0x8b,0x9f]
