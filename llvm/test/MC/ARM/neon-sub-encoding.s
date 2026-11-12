@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vsub.i8	d16, d17, d16
	vsub.i16	d16, d17, d16
	vsub.i32	d16, d17, d16
	vsub.i64	d16, d17, d16
	vsub.f32	d16, d16, d17
	vsub.i8	q8, q8, q9
	vsub.i16	q8, q8, q9
	vsub.i32	q8, q8, q9
	vsub.i64	q8, q8, q9
	vsub.f32	q8, q8, q9

	vsub.i8  d13, d21
	vsub.i16 d14, d22
	vsub.i32 d15, d23
	vsub.i64 d16, d24
	vsub.f32 d17, d25
	vsub.i8  q1, q10
	vsub.i16 q2, q9
	vsub.i32 q3, q8
	vsub.i64 q4, q7
	vsub.f32 q5, q6

@ CHECK: vsub.i8	d16, d17, d16   @ encoding: [0xa0,0x08,0x41,0xf3]
@ CHECK: vsub.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf3]
@ CHECK: vsub.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf3]
@ CHECK: vsub.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf3]
@ CHECK: vsub.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xf2]
@ CHECK: vsub.i8	q8, q8, q9      @ encoding: [0xe2,0x08,0x40,0xf3]
@ CHECK: vsub.i16	q8, q8, q9      @ encoding: [0xe2,0x08,0x50,0xf3]
@ CHECK: vsub.i32	q8, q8, q9      @ encoding: [0xe2,0x08,0x60,0xf3]
@ CHECK: vsub.i64	q8, q8, q9      @ encoding: [0xe2,0x08,0x70,0xf3]
@ CHECK: vsub.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xf2]

@ CHECK: vsub.i8	d13, d13, d21   @ encoding: [0x25,0xd8,0x0d,0xf3]
@ CHECK: vsub.i16	d14, d14, d22   @ encoding: [0x26,0xe8,0x1e,0xf3]
@ CHECK: vsub.i32	d15, d15, d23   @ encoding: [0x27,0xf8,0x2f,0xf3]
@ CHECK: vsub.i64	d16, d16, d24   @ encoding: [0xa8,0x08,0x70,0xf3]
@ CHECK: vsub.f32	d17, d17, d25   @ encoding: [0xa9,0x1d,0x61,0xf2]
@ CHECK: vsub.i8	q1, q1, q10     @ encoding: [0x64,0x28,0x02,0xf3]
@ CHECK: vsub.i16	q2, q2, q9      @ encoding: [0x62,0x48,0x14,0xf3]
@ CHECK: vsub.i32	q3, q3, q8      @ encoding: [0x60,0x68,0x26,0xf3]
@ CHECK: vsub.i64	q4, q4, q7      @ encoding: [0x4e,0x88,0x38,0xf3]
@ CHECK: vsub.f32	q5, q5, q6      @ encoding: [0x4c,0xad,0x2a,0xf2]



@ CHECK: vsubl.s8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf2]
	vsubl.s8	q8, d17, d16
@ CHECK: vsubl.s16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf2]
	vsubl.s16	q8, d17, d16
@ CHECK: vsubl.s32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf2]
	vsubl.s32	q8, d17, d16
@ CHECK: vsubl.u8	q8, d17, d16    @ encoding: [0xa0,0x02,0xc1,0xf3]
	vsubl.u8	q8, d17, d16
@ CHECK: vsubl.u16	q8, d17, d16    @ encoding: [0xa0,0x02,0xd1,0xf3]
	vsubl.u16	q8, d17, d16
@ CHECK: vsubl.u32	q8, d17, d16    @ encoding: [0xa0,0x02,0xe1,0xf3]
	vsubl.u32	q8, d17, d16
@ CHECK: vsubw.s8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf2]
	vsubw.s8	q8, q8, d18
@ CHECK: vsubw.s16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf2]
	vsubw.s16	q8, q8, d18
@ CHECK: vsubw.s32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf2]
	vsubw.s32	q8, q8, d18
@ CHECK: vsubw.u8	q8, q8, d18     @ encoding: [0xa2,0x03,0xc0,0xf3]
	vsubw.u8	q8, q8, d18
@ CHECK: vsubw.u16	q8, q8, d18     @ encoding: [0xa2,0x03,0xd0,0xf3]
	vsubw.u16	q8, q8, d18
@ CHECK: vsubw.u32	q8, q8, d18     @ encoding: [0xa2,0x03,0xe0,0xf3]
	vsubw.u32	q8, q8, d18
@ CHECK: vhsub.s8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf2]
	vhsub.s8	d16, d16, d17
@ CHECK: vhsub.s16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf2]
	vhsub.s16	d16, d16, d17
@ CHECK: vhsub.s32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf2]
	vhsub.s32	d16, d16, d17
@ CHECK: vhsub.u8	d16, d16, d17   @ encoding: [0xa1,0x02,0x40,0xf3]
	vhsub.u8	d16, d16, d17
@ CHECK: vhsub.u16	d16, d16, d17   @ encoding: [0xa1,0x02,0x50,0xf3]
	vhsub.u16	d16, d16, d17
@ CHECK: vhsub.u32	d16, d16, d17   @ encoding: [0xa1,0x02,0x60,0xf3]
	vhsub.u32	d16, d16, d17
@ CHECK: vhsub.s8	q8, q8, q9      @ encoding: [0xe2,0x02,0x40,0xf2]
	vhsub.s8	q8, q8, q9
@ CHECK: vhsub.s16	q8, q8, q9      @ encoding: [0xe2,0x02,0x50,0xf2]
	vhsub.s16	q8, q8, q9
@ CHECK: vhsub.s32	q8, q8, q9      @ encoding: [0xe2,0x02,0x60,0xf2]
	vhsub.s32	q8, q8, q9
@ CHECK: vqsub.s8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf2]
	vqsub.s8	d16, d16, d17
@ CHECK: vqsub.s16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf2]
	vqsub.s16	d16, d16, d17
@ CHECK: vqsub.s32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf2]
	vqsub.s32	d16, d16, d17
@ CHECK: vqsub.s64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf2]
	vqsub.s64	d16, d16, d17
@ CHECK: vqsub.u8	d16, d16, d17   @ encoding: [0xb1,0x02,0x40,0xf3]
	vqsub.u8	d16, d16, d17
@ CHECK: vqsub.u16	d16, d16, d17   @ encoding: [0xb1,0x02,0x50,0xf3]
	vqsub.u16	d16, d16, d17
@ CHECK: vqsub.u32	d16, d16, d17   @ encoding: [0xb1,0x02,0x60,0xf3]
	vqsub.u32	d16, d16, d17
@ CHECK: vqsub.u64	d16, d16, d17   @ encoding: [0xb1,0x02,0x70,0xf3]
	vqsub.u64	d16, d16, d17
@ CHECK: vqsub.s8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf2]
	vqsub.s8	q8, q8, q9
@ CHECK: vqsub.s16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf2]
	vqsub.s16	q8, q8, q9
@ CHECK: vqsub.s32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf2]
	vqsub.s32	q8, q8, q9
@ CHECK: vqsub.s64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf2]
	vqsub.s64	q8, q8, q9
@ CHECK: vqsub.u8	q8, q8, q9      @ encoding: [0xf2,0x02,0x40,0xf3]
	vqsub.u8	q8, q8, q9
@ CHECK: vqsub.u16	q8, q8, q9      @ encoding: [0xf2,0x02,0x50,0xf3]
	vqsub.u16	q8, q8, q9
@ CHECK: vqsub.u32	q8, q8, q9      @ encoding: [0xf2,0x02,0x60,0xf3]
	vqsub.u32	q8, q8, q9
@ CHECK: vqsub.u64	q8, q8, q9      @ encoding: [0xf2,0x02,0x70,0xf3]
	vqsub.u64	q8, q8, q9
@ CHECK: vsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf2]
	vsubhn.i16	d16, q8, q9
@ CHECK: vsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf2]
	vsubhn.i32	d16, q8, q9
@ CHECK: vsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf2]
	vsubhn.i64	d16, q8, q9
@ CHECK: vrsubhn.i16	d16, q8, q9     @ encoding: [0xa2,0x06,0xc0,0xf3]
	vrsubhn.i16	d16, q8, q9
@ CHECK: vrsubhn.i32	d16, q8, q9     @ encoding: [0xa2,0x06,0xd0,0xf3]
	vrsubhn.i32	d16, q8, q9
@ CHECK: vrsubhn.i64	d16, q8, q9     @ encoding: [0xa2,0x06,0xe0,0xf3]
	vrsubhn.i64	d16, q8, q9

	vhsub.s8	d11, d24
	vhsub.s16	d12, d23
	vhsub.s32	d13, d22
	vhsub.u8	d14, d21
	vhsub.u16	d15, d20
	vhsub.u32	d16, d19
	vhsub.s8	q1, q12
	vhsub.s16	q2, q11
	vhsub.s32	q3, q10
	vhsub.u8	q4, q9
	vhsub.u16	q5, q8
	vhsub.u32	q6, q7

@ CHECK: vhsub.s8	d11, d11, d24   @ encoding: [0x28,0xb2,0x0b,0xf2]
@ CHECK: vhsub.s16	d12, d12, d23   @ encoding: [0x27,0xc2,0x1c,0xf2]
@ CHECK: vhsub.s32	d13, d13, d22   @ encoding: [0x26,0xd2,0x2d,0xf2]
@ CHECK: vhsub.u8	d14, d14, d21   @ encoding: [0x25,0xe2,0x0e,0xf3]
@ CHECK: vhsub.u16	d15, d15, d20   @ encoding: [0x24,0xf2,0x1f,0xf3]
@ CHECK: vhsub.u32	d16, d16, d19   @ encoding: [0xa3,0x02,0x60,0xf3]
@ CHECK: vhsub.s8	q1, q1, q12     @ encoding: [0x68,0x22,0x02,0xf2]
@ CHECK: vhsub.s16	q2, q2, q11     @ encoding: [0x66,0x42,0x14,0xf2]
@ CHECK: vhsub.s32	q3, q3, q10     @ encoding: [0x64,0x62,0x26,0xf2]
@ CHECK: vhsub.u8	q4, q4, q9      @ encoding: [0x62,0x82,0x08,0xf3]
@ CHECK: vhsub.u16	q5, q5, q8      @ encoding: [0x60,0xa2,0x1a,0xf3]
@ CHECK: vhsub.u32	q6, q6, q7      @ encoding: [0x4e,0xc2,0x2c,0xf3]


	vsubw.s8  q6, d5
	vsubw.s16 q7, d1
	vsubw.s32 q8, d2
	vsubw.u8  q6, d5
	vsubw.u16 q7, d1
	vsubw.u32 q8, d2

@ CHECK: vsubw.s8	q6, q6, d5      @ encoding: [0x05,0xc3,0x8c,0xf2]
@ CHECK: vsubw.s16	q7, q7, d1      @ encoding: [0x01,0xe3,0x9e,0xf2]
@ CHECK: vsubw.s32	q8, q8, d2      @ encoding: [0x82,0x03,0xe0,0xf2]
@ CHECK: vsubw.u8	q6, q6, d5      @ encoding: [0x05,0xc3,0x8c,0xf3]
@ CHECK: vsubw.u16	q7, q7, d1      @ encoding: [0x01,0xe3,0x9e,0xf3]
@ CHECK: vsubw.u32	q8, q8, d2      @ encoding: [0x82,0x03,0xe0,0xf3]
