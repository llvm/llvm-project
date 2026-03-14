@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

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

	vmul.i8	d16, d17
	vmul.i16	d16, d17
	vmul.i32	d16, d17
	vmul.f32	d16, d17
	vmul.i8	q8, q9
	vmul.i16	q8, q9
	vmul.i32	q8, q9
	vmul.f32	q8, q9
	vmul.p8	d16, d17
	vmul.p8	q8, q9

@ CHECK: vmul.i8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf2]
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
@ CHECK: vmul.i8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf2]
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
@ CHECK: vmul.p8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf3]
@ CHECK: vmul.p8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf3]
@ CHECK: vmul.i16	d18, d8, d0[3]  @ encoding: [0x68,0x28,0xd8,0xf2]

@ CHECK: vmul.i8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf2]
@ CHECK: vmul.i16	d16, d16, d17   @ encoding: [0xb1,0x09,0x50,0xf2]
@ CHECK: vmul.i32	d16, d16, d17   @ encoding: [0xb1,0x09,0x60,0xf2]
@ CHECK: vmul.f32	d16, d16, d17   @ encoding: [0xb1,0x0d,0x40,0xf3]
@ CHECK: vmul.i8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf2]
@ CHECK: vmul.i16	q8, q8, q9      @ encoding: [0xf2,0x09,0x50,0xf2]
@ CHECK: vmul.i32	q8, q8, q9      @ encoding: [0xf2,0x09,0x60,0xf2]
@ CHECK: vmul.f32	q8, q8, q9      @ encoding: [0xf2,0x0d,0x40,0xf3]
@ CHECK: vmul.p8	d16, d16, d17   @ encoding: [0xb1,0x09,0x40,0xf3]
@ CHECK: vmul.p8	q8, q8, q9      @ encoding: [0xf2,0x09,0x40,0xf3]


	vqdmulh.s16	d16, d16, d17
	vqdmulh.s32	d16, d16, d17
	vqdmulh.s16	q8, q8, q9
	vqdmulh.s32	q8, q8, q9
	vqdmulh.s16	d16, d17
	vqdmulh.s32	d16, d17
	vqdmulh.s16	q8, q9
	vqdmulh.s32	q8, q9
	vqdmulh.s16	d11, d2, d3[0]

@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf2]
@ CHECK: vqdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf2]
@ CHECK: vqdmulh.s16	d11, d2, d3[0]  @ encoding: [0x43,0xbc,0x92,0xf2]


	vqrdmulh.s16	d16, d16, d17
	vqrdmulh.s32	d16, d16, d17
	vqrdmulh.s16	q8, q8, q9
	vqrdmulh.s32	q8, q8, q9

@ CHECK: vqrdmulh.s16	d16, d16, d17   @ encoding: [0xa1,0x0b,0x50,0xf3]
@ CHECK: vqrdmulh.s32	d16, d16, d17   @ encoding: [0xa1,0x0b,0x60,0xf3]
@ CHECK: vqrdmulh.s16	q8, q8, q9      @ encoding: [0xe2,0x0b,0x50,0xf3]
@ CHECK: vqrdmulh.s32	q8, q8, q9      @ encoding: [0xe2,0x0b,0x60,0xf3]


	vmull.s8	q8, d16, d17
	vmull.s16	q8, d16, d17
	vmull.s32	q8, d16, d17
	vmull.u8	q8, d16, d17
	vmull.u16	q8, d16, d17
	vmull.u32	q8, d16, d17
	vmull.p8	q8, d16, d17

@ CHECK: vmull.s8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf2]
@ CHECK: vmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf2]
@ CHECK: vmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf2]
@ CHECK: vmull.u8	q8, d16, d17    @ encoding: [0xa1,0x0c,0xc0,0xf3]
@ CHECK: vmull.u16	q8, d16, d17    @ encoding: [0xa1,0x0c,0xd0,0xf3]
@ CHECK: vmull.u32	q8, d16, d17    @ encoding: [0xa1,0x0c,0xe0,0xf3]
@ CHECK: vmull.p8	q8, d16, d17    @ encoding: [0xa1,0x0e,0xc0,0xf2]


	vqdmull.s16	q8, d16, d17
	vqdmull.s32	q8, d16, d17

@ CHECK: vqdmull.s16	q8, d16, d17    @ encoding: [0xa1,0x0d,0xd0,0xf2]
@ CHECK: vqdmull.s32	q8, d16, d17    @ encoding: [0xa1,0x0d,0xe0,0xf2]


        vmul.i16 d0, d4[2]
        vmul.s16 d1, d7[3]
        vmul.u16 d2, d1[1]
        vmul.i32 d3, d2[0]
        vmul.s32 d4, d3[1]
        vmul.u32 d5, d4[0]
        vmul.f32 d6, d5[1]

        vmul.i16 q0, d4[2]
        vmul.s16 q1, d7[3]
        vmul.u16 q2, d1[1]
        vmul.i32 q3, d2[0]
        vmul.s32 q4, d3[1]
        vmul.u32 q5, d4[0]
        vmul.f32 q6, d5[1]

        vmul.i16 d9, d0, d4[2]
        vmul.s16 d8, d1, d7[3]
        vmul.u16 d7, d2, d1[1]
        vmul.i32 d6, d3, d2[0]
        vmul.s32 d5, d4, d3[1]
        vmul.u32 d4, d5, d4[0]
        vmul.f32 d3, d6, d5[1]

        vmul.i16 q9, q0, d4[2]
        vmul.s16 q8, q1, d7[3]
        vmul.u16 q7, q2, d1[1]
        vmul.i32 q6, q3, d2[0]
        vmul.s32 q5, q4, d3[1]
        vmul.u32 q4, q5, d4[0]
        vmul.f32 q3, q6, d5[1]

@ CHECK: vmul.i16	d0, d0, d4[2]   @ encoding: [0x64,0x08,0x90,0xf2]
@ CHECK: vmul.i16	d1, d1, d7[3]   @ encoding: [0x6f,0x18,0x91,0xf2]
@ CHECK: vmul.i16	d2, d2, d1[1]   @ encoding: [0x49,0x28,0x92,0xf2]
@ CHECK: vmul.i32	d3, d3, d2[0]   @ encoding: [0x42,0x38,0xa3,0xf2]
@ CHECK: vmul.i32	d4, d4, d3[1]   @ encoding: [0x63,0x48,0xa4,0xf2]
@ CHECK: vmul.i32	d5, d5, d4[0]   @ encoding: [0x44,0x58,0xa5,0xf2]
@ CHECK: vmul.f32	d6, d6, d5[1]   @ encoding: [0x65,0x69,0xa6,0xf2]

@ CHECK: vmul.i16	q0, q0, d4[2]   @ encoding: [0x64,0x08,0x90,0xf3]
@ CHECK: vmul.i16	q1, q1, d7[3]   @ encoding: [0x6f,0x28,0x92,0xf3]
@ CHECK: vmul.i16	q2, q2, d1[1]   @ encoding: [0x49,0x48,0x94,0xf3]
@ CHECK: vmul.i32	q3, q3, d2[0]   @ encoding: [0x42,0x68,0xa6,0xf3]
@ CHECK: vmul.i32	q4, q4, d3[1]   @ encoding: [0x63,0x88,0xa8,0xf3]
@ CHECK: vmul.i32	q5, q5, d4[0]   @ encoding: [0x44,0xa8,0xaa,0xf3]
@ CHECK: vmul.f32	q6, q6, d5[1]   @ encoding: [0x65,0xc9,0xac,0xf3]

@ CHECK: vmul.i16	d9, d0, d4[2]   @ encoding: [0x64,0x98,0x90,0xf2]
@ CHECK: vmul.i16	d8, d1, d7[3]   @ encoding: [0x6f,0x88,0x91,0xf2]
@ CHECK: vmul.i16	d7, d2, d1[1]   @ encoding: [0x49,0x78,0x92,0xf2]
@ CHECK: vmul.i32	d6, d3, d2[0]   @ encoding: [0x42,0x68,0xa3,0xf2]
@ CHECK: vmul.i32	d5, d4, d3[1]   @ encoding: [0x63,0x58,0xa4,0xf2]
@ CHECK: vmul.i32	d4, d5, d4[0]   @ encoding: [0x44,0x48,0xa5,0xf2]
@ CHECK: vmul.f32	d3, d6, d5[1]   @ encoding: [0x65,0x39,0xa6,0xf2]

@ CHECK: vmul.i16	q9, q0, d4[2]   @ encoding: [0x64,0x28,0xd0,0xf3]
@ CHECK: vmul.i16	q8, q1, d7[3]   @ encoding: [0x6f,0x08,0xd2,0xf3]
@ CHECK: vmul.i16	q7, q2, d1[1]   @ encoding: [0x49,0xe8,0x94,0xf3]
@ CHECK: vmul.i32	q6, q3, d2[0]   @ encoding: [0x42,0xc8,0xa6,0xf3]
@ CHECK: vmul.i32	q5, q4, d3[1]   @ encoding: [0x63,0xa8,0xa8,0xf3]
@ CHECK: vmul.i32	q4, q5, d4[0]   @ encoding: [0x44,0x88,0xaa,0xf3]
@ CHECK: vmul.f32	q3, q6, d5[1]   @ encoding: [0x65,0x69,0xac,0xf3]
