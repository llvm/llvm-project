@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s


@ CHECK: vadd.i8	d16, d17, d16           @ encoding: [0xa0,0x08,0x41,0xf2]
	vadd.i8	d16, d17, d16
@ CHECK: vadd.i16	d16, d17, d16   @ encoding: [0xa0,0x08,0x51,0xf2]
	vadd.i16	d16, d17, d16
@ CHECK: vadd.i64	d16, d17, d16   @ encoding: [0xa0,0x08,0x71,0xf2]
	vadd.i64	d16, d17, d16
@ CHECK: vadd.i32	d16, d17, d16   @ encoding: [0xa0,0x08,0x61,0xf2]
	vadd.i32	d16, d17, d16
@ CHECK: vadd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x40,0xf2]
	vadd.f32	d16, d16, d17
@ CHECK: vadd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x40,0xf2]
	vadd.f32	q8, q8, q9

@ CHECK: vaddl.s8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf2]
	vaddl.s8	q8, d17, d16
@ CHECK: vaddl.s16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf2]
	vaddl.s16	q8, d17, d16
@ CHECK: vaddl.s32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf2]
	vaddl.s32	q8, d17, d16
@ CHECK: vaddl.u8	q8, d17, d16    @ encoding: [0xa0,0x00,0xc1,0xf3]
	vaddl.u8	q8, d17, d16
@ CHECK: vaddl.u16	q8, d17, d16    @ encoding: [0xa0,0x00,0xd1,0xf3]
	vaddl.u16	q8, d17, d16
@ CHECK: vaddl.u32	q8, d17, d16    @ encoding: [0xa0,0x00,0xe1,0xf3]
	vaddl.u32	q8, d17, d16

@ CHECK: vaddw.s8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf2]
	vaddw.s8	q8, q8, d18
@ CHECK: vaddw.s16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf2]
	vaddw.s16	q8, q8, d18
@ CHECK: vaddw.s32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf2]
	vaddw.s32	q8, q8, d18
@ CHECK: vaddw.u8	q8, q8, d18     @ encoding: [0xa2,0x01,0xc0,0xf3]
	vaddw.u8	q8, q8, d18
@ CHECK: vaddw.u16	q8, q8, d18     @ encoding: [0xa2,0x01,0xd0,0xf3]
	vaddw.u16	q8, q8, d18
@ CHECK: vaddw.u32	q8, q8, d18     @ encoding: [0xa2,0x01,0xe0,0xf3]
	vaddw.u32	q8, q8, d18

@ CHECK: vhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf2]
	vhadd.s8	d16, d16, d17
@ CHECK: vhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf2]
	vhadd.s16	d16, d16, d17
@ CHECK: vhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf2]
	vhadd.s32	d16, d16, d17
@ CHECK: vhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x00,0x40,0xf3]
	vhadd.u8	d16, d16, d17
@ CHECK: vhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x00,0x50,0xf3]
	vhadd.u16	d16, d16, d17
@ CHECK: vhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x00,0x60,0xf3]
	vhadd.u32	d16, d16, d17
@ CHECK: vhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf2]
	vhadd.s8	q8, q8, q9
@ CHECK: vhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf2]
	vhadd.s16	q8, q8, q9
@ CHECK: vhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf2]
	vhadd.s32	q8, q8, q9
  @ CHECK: vhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x00,0x40,0xf3]
	vhadd.u8	q8, q8, q9
@ CHECK: vhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x00,0x50,0xf3]
	vhadd.u16	q8, q8, q9
@ CHECK: vhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x00,0x60,0xf3]
	vhadd.u32	q8, q8, q9


	vhadd.s8	d11, d24
	vhadd.s16	d12, d23
	vhadd.s32	d13, d22
	vhadd.u8	d14, d21
	vhadd.u16	d15, d20
	vhadd.u32	d16, d19
	vhadd.s8	q1, q12
	vhadd.s16	q2, q11
	vhadd.s32	q3, q10
	vhadd.u8	q4, q9
	vhadd.u16	q5, q8
	vhadd.u32	q6, q7

@ CHECK: vhadd.s8	d11, d11, d24   @ encoding: [0x28,0xb0,0x0b,0xf2]
@ CHECK: vhadd.s16	d12, d12, d23   @ encoding: [0x27,0xc0,0x1c,0xf2]
@ CHECK: vhadd.s32	d13, d13, d22   @ encoding: [0x26,0xd0,0x2d,0xf2]
@ CHECK: vhadd.u8	d14, d14, d21   @ encoding: [0x25,0xe0,0x0e,0xf3]
@ CHECK: vhadd.u16	d15, d15, d20   @ encoding: [0x24,0xf0,0x1f,0xf3]
@ CHECK: vhadd.u32	d16, d16, d19   @ encoding: [0xa3,0x00,0x60,0xf3]
@ CHECK: vhadd.s8	q1, q1, q12     @ encoding: [0x68,0x20,0x02,0xf2]
@ CHECK: vhadd.s16	q2, q2, q11     @ encoding: [0x66,0x40,0x14,0xf2]
@ CHECK: vhadd.s32	q3, q3, q10     @ encoding: [0x64,0x60,0x26,0xf2]
@ CHECK: vhadd.u8	q4, q4, q9      @ encoding: [0x62,0x80,0x08,0xf3]
@ CHECK: vhadd.u16	q5, q5, q8      @ encoding: [0x60,0xa0,0x1a,0xf3]
@ CHECK: vhadd.u32	q6, q6, q7      @ encoding: [0x4e,0xc0,0x2c,0xf3]

	vrhadd.s8	d16, d16, d17
	vrhadd.s16	d16, d16, d17
	vrhadd.s32	d16, d16, d17
	vrhadd.u8	d16, d16, d17
	vrhadd.u16	d16, d16, d17
	vrhadd.u32	d16, d16, d17
	vrhadd.s8	q8, q8, q9
	vrhadd.s16	q8, q8, q9
	vrhadd.s32	q8, q8, q9
	vrhadd.u8	q8, q8, q9
	vrhadd.u16	q8, q8, q9
	vrhadd.u32	q8, q8, q9
        @ Two-operand forms.
	vrhadd.s8	d16, d17
	vrhadd.s16	d16, d17
	vrhadd.s32	d16, d17
	vrhadd.u8	d16, d17
	vrhadd.u16	d16, d17
	vrhadd.u32	d16, d17
	vrhadd.s8	q8, q9
	vrhadd.s16	q8, q9
	vrhadd.s32	q8, q9
	vrhadd.u8	q8, q9
	vrhadd.u16	q8, q9
	vrhadd.u32	q8, q9

@ CHECK: vrhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf2]
@ CHECK: vrhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf2]
@ CHECK: vrhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf2]
@ CHECK: vrhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf3]
@ CHECK: vrhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf3]
@ CHECK: vrhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf3]
@ CHECK: vrhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf2]
@ CHECK: vrhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf2]
@ CHECK: vrhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf2]
@ CHECK: vrhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf3]
@ CHECK: vrhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf3]
@ CHECK: vrhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf3]

@ CHECK: vrhadd.s8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf2]
@ CHECK: vrhadd.s16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf2]
@ CHECK: vrhadd.s32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf2]
@ CHECK: vrhadd.u8	d16, d16, d17   @ encoding: [0xa1,0x01,0x40,0xf3]
@ CHECK: vrhadd.u16	d16, d16, d17   @ encoding: [0xa1,0x01,0x50,0xf3]
@ CHECK: vrhadd.u32	d16, d16, d17   @ encoding: [0xa1,0x01,0x60,0xf3]
@ CHECK: vrhadd.s8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf2]
@ CHECK: vrhadd.s16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf2]
@ CHECK: vrhadd.s32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf2]
@ CHECK: vrhadd.u8	q8, q8, q9      @ encoding: [0xe2,0x01,0x40,0xf3]
@ CHECK: vrhadd.u16	q8, q8, q9      @ encoding: [0xe2,0x01,0x50,0xf3]
@ CHECK: vrhadd.u32	q8, q8, q9      @ encoding: [0xe2,0x01,0x60,0xf3]


	vqadd.s8	d16, d16, d17
	vqadd.s16	d16, d16, d17
	vqadd.s32	d16, d16, d17
	vqadd.s64	d16, d16, d17
	vqadd.u8	d16, d16, d17
	vqadd.u16	d16, d16, d17
	vqadd.u32	d16, d16, d17
	vqadd.u64	d16, d16, d17

@ CHECK: vqadd.s8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf2]
@ CHECK: vqadd.s16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf2]
@ CHECK: vqadd.s32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf2]
@ CHECK: vqadd.s64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf2]
@ CHECK: vqadd.u8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf3]
@ CHECK: vqadd.u16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf3]
@ CHECK: vqadd.u32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf3]
@ CHECK: vqadd.u64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf3]

	vqadd.s8	q8, q8, q9
	vqadd.s16	q8, q8, q9
	vqadd.s32	q8, q8, q9
	vqadd.s64	q8, q8, q9
	vqadd.u8	q8, q8, q9
	vqadd.u16	q8, q8, q9
	vqadd.u32	q8, q8, q9
	vqadd.u64	q8, q8, q9

@ CHECK: vqadd.s8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf2]
@ CHECK: vqadd.s16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf2]
@ CHECK: vqadd.s32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf2]
@ CHECK: vqadd.s64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf2]
@ CHECK: vqadd.u8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf3]
@ CHECK: vqadd.u16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf3]
@ CHECK: vqadd.u32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf3]
@ CHECK: vqadd.u64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf3]


@ two-operand variants.
	vqadd.s8	d16, d17
	vqadd.s16	d16, d17
	vqadd.s32	d16, d17
	vqadd.s64	d16, d17
	vqadd.u8	d16, d17
	vqadd.u16	d16, d17
	vqadd.u32	d16, d17
	vqadd.u64	d16, d17

@ CHECK: vqadd.s8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf2]
@ CHECK: vqadd.s16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf2]
@ CHECK: vqadd.s32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf2]
@ CHECK: vqadd.s64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf2]
@ CHECK: vqadd.u8	d16, d16, d17   @ encoding: [0xb1,0x00,0x40,0xf3]
@ CHECK: vqadd.u16	d16, d16, d17   @ encoding: [0xb1,0x00,0x50,0xf3]
@ CHECK: vqadd.u32	d16, d16, d17   @ encoding: [0xb1,0x00,0x60,0xf3]
@ CHECK: vqadd.u64	d16, d16, d17   @ encoding: [0xb1,0x00,0x70,0xf3]

	vqadd.s8	q8, q9
	vqadd.s16	q8, q9
	vqadd.s32	q8, q9
	vqadd.s64	q8, q9
	vqadd.u8	q8, q9
	vqadd.u16	q8, q9
	vqadd.u32	q8, q9
	vqadd.u64	q8, q9

@ CHECK: vqadd.s8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf2]
@ CHECK: vqadd.s16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf2]
@ CHECK: vqadd.s32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf2]
@ CHECK: vqadd.s64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf2]
@ CHECK: vqadd.u8	q8, q8, q9      @ encoding: [0xf2,0x00,0x40,0xf3]
@ CHECK: vqadd.u16	q8, q8, q9      @ encoding: [0xf2,0x00,0x50,0xf3]
@ CHECK: vqadd.u32	q8, q8, q9      @ encoding: [0xf2,0x00,0x60,0xf3]
@ CHECK: vqadd.u64	q8, q8, q9      @ encoding: [0xf2,0x00,0x70,0xf3]


@ CHECK: vaddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf2]
	vaddhn.i16	d16, q8, q9
@ CHECK: vaddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf2]
	vaddhn.i32	d16, q8, q9
@ CHECK: vaddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf2]
	vaddhn.i64	d16, q8, q9
@ CHECK: vraddhn.i16	d16, q8, q9     @ encoding: [0xa2,0x04,0xc0,0xf3]
	vraddhn.i16	d16, q8, q9
@ CHECK: vraddhn.i32	d16, q8, q9     @ encoding: [0xa2,0x04,0xd0,0xf3]
	vraddhn.i32	d16, q8, q9
@ CHECK: vraddhn.i64	d16, q8, q9     @ encoding: [0xa2,0x04,0xe0,0xf3]
	vraddhn.i64	d16, q8, q9


@ Two-operand variants

	vadd.i8  d6, d5
	vadd.i16 d7, d1
	vadd.i32 d8, d2
	vadd.i64 d9, d3

	vadd.i8  q6, q5
	vadd.i16 q7, q1
	vadd.i32 q8, q2
	vadd.i64 q9, q3

@ CHECK: vadd.i8	d6, d6, d5      @ encoding: [0x05,0x68,0x06,0xf2]
@ CHECK: vadd.i16	d7, d7, d1      @ encoding: [0x01,0x78,0x17,0xf2]
@ CHECK: vadd.i32	d8, d8, d2      @ encoding: [0x02,0x88,0x28,0xf2]
@ CHECK: vadd.i64	d9, d9, d3      @ encoding: [0x03,0x98,0x39,0xf2]

@ CHECK: vadd.i8	q6, q6, q5      @ encoding: [0x4a,0xc8,0x0c,0xf2]
@ CHECK: vadd.i16	q7, q7, q1      @ encoding: [0x42,0xe8,0x1e,0xf2]
@ CHECK: vadd.i32	q8, q8, q2      @ encoding: [0xc4,0x08,0x60,0xf2]
@ CHECK: vadd.i64	q9, q9, q3      @ encoding: [0xc6,0x28,0x72,0xf2]


	vaddw.s8  q6, d5
	vaddw.s16 q7, d1
	vaddw.s32 q8, d2

	vaddw.u8  q6, d5
	vaddw.u16 q7, d1
	vaddw.u32 q8, d2

@ CHECK: vaddw.s8	q6, q6, d5      @ encoding: [0x05,0xc1,0x8c,0xf2]
@ CHECK: vaddw.s16	q7, q7, d1      @ encoding: [0x01,0xe1,0x9e,0xf2]
@ CHECK: vaddw.s32	q8, q8, d2      @ encoding: [0x82,0x01,0xe0,0xf2]

@ CHECK: vaddw.u8	q6, q6, d5      @ encoding: [0x05,0xc1,0x8c,0xf3]
@ CHECK: vaddw.u16	q7, q7, d1      @ encoding: [0x01,0xe1,0x9e,0xf3]
@ CHECK: vaddw.u32	q8, q8, d2      @ encoding: [0x82,0x01,0xe0,0xf3]
