@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vmul.i8	d16, d16, d17
	vmul.i16	d16, d16, d17
	vmul.i32	d16, d16, d17
	vmul.f32	d16, d16, d17
	vmul.i8	q8, q8, q9
	vmul.i16	q8, q8, q9
	vmul.i32	q8, q8, q9
	vmul.f32	q8, q8, q9
	vmul.p8	d16, d16, d17
	vmul.p8	q8, q8, q9
	vmul.i16	d18, d8, d0[3]

@ CHECK: vmul.i8	d16, d16, d17   @ encoding: [0x40,0xef,0xb1,0x09]
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0x50,0xef,0xb1,0x09]
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0x60,0xef,0xb1,0x09]
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0x40,0xff,0xb1,0x0d]
@ CHECK: vmul.i8	q8, q8, q9      @ encoding: [0x40,0xef,0xf2,0x09]
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0x50,0xef,0xf2,0x09]
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0x60,0xef,0xf2,0x09]
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0x40,0xff,0xf2,0x0d]
@ CHECK: vmul.p8	d16, d16, d17   @ encoding: [0x40,0xff,0xb1,0x09]
@ CHECK: vmul.p8	q8, q8, q9      @ encoding: [0x40,0xff,0xf2,0x09]
@ CHECK: vmul.i16	d18, d8, d0[3]  @ encoding: [0xd8,0xef,0x68,0x28]


	vqdmulh.s16	d16, d16, d17
	vqdmulh.s32	d16, d16, d17
	vqdmulh.s16	q8, q8, q9
	vqdmulh.s32	q8, q8, q9
	vqdmulh.s16	d11, d2, d3[0]

@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0x50,0xef,0xa1,0x0b]
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0x60,0xef,0xa1,0x0b]
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0x50,0xef,0xe2,0x0b]
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x0b]
@ CHECK: vqdmulh.s16	d11, d2, d3[0]  @ encoding: [0x92,0xef,0x43,0xbc]


	vqrdmulh.s16	d16, d16, d17
	vqrdmulh.s32	d16, d16, d17
	vqrdmulh.s16	q8, q8, q9
	vqrdmulh.s32	q8, q8, q9

@ CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0x50,0xff,0xa1,0x0b]
@ CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0x60,0xff,0xa1,0x0b]
@ CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0x50,0xff,0xe2,0x0b]
@ CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0x60,0xff,0xe2,0x0b]


	vmull.s8	q8, d16, d17
	vmull.s16	q8, d16, d17
	vmull.s32	q8, d16, d17
	vmull.u8	q8, d16, d17
	vmull.u16	q8, d16, d17
	vmull.u32	q8, d16, d17
	vmull.p8	q8, d16, d17

@ CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xc0,0xef,0xa1,0x0c]
@ CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xd0,0xef,0xa1,0x0c]
@ CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x0c]
@ CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xc0,0xff,0xa1,0x0c]
@ CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xd0,0xff,0xa1,0x0c]
@ CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xe0,0xff,0xa1,0x0c]
@ CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xc0,0xef,0xa1,0x0e]


	vqdmull.s16	q8, d16, d17
	vqdmull.s32	q8, d16, d17
        vqdmull.s16	q1, d7, d1[1]

@ CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xd0,0xef,0xa1,0x0d]
@ CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x0d]
@ CHECK: vqdmull.s16	q1, d7, d1[1]   @ encoding: [0x97,0xef,0x49,0x2b]

