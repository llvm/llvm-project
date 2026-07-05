@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vabd.s8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xf2]
	vabd.s8	d16, d16, d17
@ CHECK: vabd.s16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xf2]
	vabd.s16	d16, d16, d17
@ CHECK: vabd.s32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xf2]
	vabd.s32	d16, d16, d17
@ CHECK: vabd.u8	d16, d16, d17           @ encoding: [0xa1,0x07,0x40,0xf3]
	vabd.u8	d16, d16, d17
@ CHECK: vabd.u16	d16, d16, d17   @ encoding: [0xa1,0x07,0x50,0xf3]
	vabd.u16	d16, d16, d17
  @ CHECK: vabd.u32	d16, d16, d17   @ encoding: [0xa1,0x07,0x60,0xf3]
	vabd.u32	d16, d16, d17
@ CHECK: vabd.f32	d16, d16, d17   @ encoding: [0xa1,0x0d,0x60,0xf3]
	vabd.f32	d16, d16, d17
@ CHECK: vabd.s8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xf2]
	vabd.s8	q8, q8, q9
@ CHECK: vabd.s16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xf2]
	vabd.s16	q8, q8, q9
@ CHECK: vabd.s32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xf2]
	vabd.s32	q8, q8, q9
@ CHECK: vabd.u8	q8, q8, q9              @ encoding: [0xe2,0x07,0x40,0xf3]
	vabd.u8	q8, q8, q9
@ CHECK: vabd.u16	q8, q8, q9      @ encoding: [0xe2,0x07,0x50,0xf3]
	vabd.u16	q8, q8, q9
@ CHECK: vabd.u32	q8, q8, q9      @ encoding: [0xe2,0x07,0x60,0xf3]
	vabd.u32	q8, q8, q9
@ CHECK: vabd.f32	q8, q8, q9      @ encoding: [0xe2,0x0d,0x60,0xf3]
	vabd.f32	q8, q8, q9

@ CHECK: vabdl.s8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xf2]
	vabdl.s8	q8, d16, d17
@ CHECK: vabdl.s16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xf2]
	vabdl.s16	q8, d16, d17
@ CHECK: vabdl.s32	q8, d16, d17    @ encoding: [0xa1,0x07,0xe0,0xf2]
	vabdl.s32	q8, d16, d17
@ CHECK: vabdl.u8	q8, d16, d17    @ encoding: [0xa1,0x07,0xc0,0xf3]
	vabdl.u8	q8, d16, d17
@ CHECK: vabdl.u16	q8, d16, d17    @ encoding: [0xa1,0x07,0xd0,0xf3]
	vabdl.u16	q8, d16, d17
@ CHECK: vabdl.u32	q8, d16, d17    @ encoding: [0xa1,0x07,0xe0,0xf3]
	vabdl.u32	q8, d16, d17

@ CHECK: vaba.s8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xf2]
	vaba.s8	d16, d18, d17
@ CHECK: vaba.s16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xf2]
	vaba.s16	d16, d18, d17
@ CHECK: vaba.s32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xf2]
	vaba.s32	d16, d18, d17
@ CHECK: vaba.u8	d16, d18, d17           @ encoding: [0xb1,0x07,0x42,0xf3]
	vaba.u8	d16, d18, d17
@ CHECK: vaba.u16	d16, d18, d17   @ encoding: [0xb1,0x07,0x52,0xf3]
	vaba.u16	d16, d18, d17
@ CHECK: vaba.u32	d16, d18, d17   @ encoding: [0xb1,0x07,0x62,0xf3]
	vaba.u32	d16, d18, d17
@ CHECK: vaba.s8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xf2]
	vaba.s8	q9, q8, q10
@ CHECK: vaba.s16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xf2]
	vaba.s16	q9, q8, q10
@ CHECK: vaba.s32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xf2]
	vaba.s32	q9, q8, q10
@ CHECK: vaba.u8	q9, q8, q10             @ encoding: [0xf4,0x27,0x40,0xf3]
	vaba.u8	q9, q8, q10
@ CHECK: vaba.u16	q9, q8, q10     @ encoding: [0xf4,0x27,0x50,0xf3]
	vaba.u16	q9, q8, q10
@ CHECK: vaba.u32	q9, q8, q10     @ encoding: [0xf4,0x27,0x60,0xf3]
	vaba.u32	q9, q8, q10

@ CHECK: vabal.s8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xf2]
	vabal.s8	q8, d19, d18
@ CHECK: vabal.s16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xf2]
	vabal.s16	q8, d19, d18
@ CHECK: vabal.s32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xf2]
	vabal.s32	q8, d19, d18
@ CHECK: vabal.u8	q8, d19, d18    @ encoding: [0xa2,0x05,0xc3,0xf3]
	vabal.u8	q8, d19, d18
@ CHECK: 	vabal.u16	q8, d19, d18    @ encoding: [0xa2,0x05,0xd3,0xf3]
	vabal.u16	q8, d19, d18
@ CHECK: vabal.u32	q8, d19, d18    @ encoding: [0xa2,0x05,0xe3,0xf3]
	vabal.u32	q8, d19, d18

