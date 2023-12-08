@RUN: llvm-mc -triple thumbv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s

.code 16

	vabd.s8	d16, d16, d17
	vabd.s16	d16, d16, d17
	vabd.s32	d16, d16, d17
	vabd.u8	d16, d16, d17
	vabd.u16	d16, d16, d17
	vabd.u32	d16, d16, d17
	vabd.f32	d16, d16, d17
	vabd.s8	q8, q8, q9
	vabd.s16	q8, q8, q9
	vabd.s32	q8, q8, q9
	vabd.u8	q8, q8, q9
	vabd.u16	q8, q8, q9
	vabd.u32	q8, q8, q9
	vabd.f32	q8, q8, q9

@ CHECK: vabd.s8	d16, d16, d17   @ encoding: [0x40,0xef,0xa1,0x07]
@ CHECK: vabd.s16	d16, d16, d17   @ encoding: [0x50,0xef,0xa1,0x07]
@ CHECK: vabd.s32	d16, d16, d17   @ encoding: [0x60,0xef,0xa1,0x07]
@ CHECK: vabd.u8	d16, d16, d17   @ encoding: [0x40,0xff,0xa1,0x07]
@ CHECK: vabd.u16	d16, d16, d17   @ encoding: [0x50,0xff,0xa1,0x07]
@ CHECK: vabd.u32	d16, d16, d17   @ encoding: [0x60,0xff,0xa1,0x07]
@ CHECK: vabd.f32	d16, d16, d17   @ encoding: [0x60,0xff,0xa1,0x0d]
@ CHECK: vabd.s8	q8, q8, q9      @ encoding: [0x40,0xef,0xe2,0x07]
@ CHECK: vabd.s16	q8, q8, q9      @ encoding: [0x50,0xef,0xe2,0x07]
@ CHECK: vabd.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x07]
@ CHECK: vabd.u8	q8, q8, q9      @ encoding: [0x40,0xff,0xe2,0x07]
@ CHECK: vabd.u16	q8, q8, q9      @ encoding: [0x50,0xff,0xe2,0x07]
@ CHECK: vabd.u32	q8, q8, q9      @ encoding: [0x60,0xff,0xe2,0x07]
@ CHECK: vabd.f32	q8, q8, q9      @ encoding: [0x60,0xff,0xe2,0x0d]


	vabdl.s8	q8, d16, d17
	vabdl.s16	q8, d16, d17
	vabdl.s32	q8, d16, d17
	vabdl.u8	q8, d16, d17
	vabdl.u16	q8, d16, d17
	vabdl.u32	q8, d16, d17

@ CHECK: vabdl.s8	q8, d16, d17    @ encoding: [0xc0,0xef,0xa1,0x07]
@ CHECK: vabdl.s16	q8, d16, d17    @ encoding: [0xd0,0xef,0xa1,0x07]
@ CHECK: vabdl.s32	q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x07]
@ CHECK: vabdl.u8	q8, d16, d17    @ encoding: [0xc0,0xff,0xa1,0x07]
@ CHECK: vabdl.u16	q8, d16, d17    @ encoding: [0xd0,0xff,0xa1,0x07]
@ CHECK: vabdl.u32	q8, d16, d17    @ encoding: [0xe0,0xff,0xa1,0x07]


	vaba.s8	d16, d18, d17
	vaba.s16	d16, d18, d17
	vaba.s32	d16, d18, d17
	vaba.u8	d16, d18, d17
	vaba.u16	d16, d18, d17
	vaba.u32	d16, d18, d17
	vaba.s8	q9, q8, q10
	vaba.s16	q9, q8, q10
	vaba.s32	q9, q8, q10
	vaba.u8	q9, q8, q10
	vaba.u16	q9, q8, q10
	vaba.u32	q9, q8, q10

@ CHECK: vaba.s8	d16, d18, d17   @ encoding: [0x42,0xef,0xb1,0x07]
@ CHECK: vaba.s16	d16, d18, d17   @ encoding: [0x52,0xef,0xb1,0x07]
@ CHECK: vaba.s32	d16, d18, d17   @ encoding: [0x62,0xef,0xb1,0x07]
@ CHECK: vaba.u8	d16, d18, d17   @ encoding: [0x42,0xff,0xb1,0x07]
@ CHECK: vaba.u16	d16, d18, d17   @ encoding: [0x52,0xff,0xb1,0x07]
@ CHECK: vaba.u32	d16, d18, d17   @ encoding: [0x62,0xff,0xb1,0x07]
@ CHECK: vaba.s8	q9, q8, q10     @ encoding: [0x40,0xef,0xf4,0x27]
@ CHECK: vaba.s16	q9, q8, q10     @ encoding: [0x50,0xef,0xf4,0x27]
@ CHECK: vaba.s32	q9, q8, q10     @ encoding: [0x60,0xef,0xf4,0x27]
@ CHECK: vaba.u8	q9, q8, q10     @ encoding: [0x40,0xff,0xf4,0x27]
@ CHECK: vaba.u16	q9, q8, q10     @ encoding: [0x50,0xff,0xf4,0x27]
@ CHECK: vaba.u32	q9, q8, q10     @ encoding: [0x60,0xff,0xf4,0x27]


	vabal.s8	q8, d19, d18
	vabal.s16	q8, d19, d18
	vabal.s32	q8, d19, d18
	vabal.u8	q8, d19, d18
	vabal.u16	q8, d19, d18
	vabal.u32	q8, d19, d18

@ CHECK: vabal.s8	q8, d19, d18    @ encoding: [0xc3,0xef,0xa2,0x05]
@ CHECK: vabal.s16	q8, d19, d18    @ encoding: [0xd3,0xef,0xa2,0x05]
@ CHECK: vabal.s32	q8, d19, d18    @ encoding: [0xe3,0xef,0xa2,0x05]
@ CHECK: vabal.u8	q8, d19, d18    @ encoding: [0xc3,0xff,0xa2,0x05]
@ CHECK: vabal.u16	q8, d19, d18    @ encoding: [0xd3,0xff,0xa2,0x05]
@ CHECK: vabal.u32	q8, d19, d18    @ encoding: [0xe3,0xff,0xa2,0x05]

