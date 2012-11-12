@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vceq.i8	d16, d16, d17
	vceq.i16	d16, d16, d17
	vceq.i32	d16, d16, d17
	vceq.f32	d16, d16, d17
	vceq.i8	q8, q8, q9
	vceq.i16	q8, q8, q9
	vceq.i32	q8, q8, q9
	vceq.f32	q8, q8, q9

@ CHECK: vceq.i8	d16, d16, d17   @ encoding: [0xb1,0x08,0x40,0xf3]
@ CHECK: vceq.i16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf3]
@ CHECK: vceq.i32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf3]
@ CHECK: vceq.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf2]
@ CHECK: vceq.i8	q8, q8, q9      @ encoding: [0xf2,0x08,0x40,0xf3]
@ CHECK: vceq.i16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf3]
@ CHECK: vceq.i32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf3]
@ CHECK: vceq.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf2]

	vcge.s8	d16, d16, d17
	vcge.s16	d16, d16, d17
	vcge.s32	d16, d16, d17
	vcge.u8	d16, d16, d17
	vcge.u16	d16, d16, d17
	vcge.u32	d16, d16, d17
	vcge.f32	d16, d16, d17
	vcge.s8	q8, q8, q9
	vcge.s16	q8, q8, q9
	vcge.s32	q8, q8, q9
	vcge.u8	q8, q8, q9
	vcge.u16	q8, q8, q9
	vcge.u32	q8, q8, q9
	vcge.f32	q8, q8, q9
	vacge.f32	d16, d16, d17
	vacge.f32	q8, q8, q9

@ CHECK: vcge.s8	d16, d16, d17   @ encoding: [0xb1,0x03,0x40,0xf2]
@ CHECK: vcge.s16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf2]
@ CHECK: vcge.s32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf2]
@ CHECK: vcge.u8	d16, d16, d17   @ encoding: [0xb1,0x03,0x40,0xf3]
@ CHECK: vcge.u16	d16, d16, d17   @ encoding: [0xb1,0x03,0x50,0xf3]
@ CHECK: vcge.u32	d16, d16, d17   @ encoding: [0xb1,0x03,0x60,0xf3]
@ CHECK: vcge.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x40,0xf3]
@ CHECK: vcge.s8	q8, q8, q9      @ encoding: [0xf2,0x03,0x40,0xf2]
@ CHECK: vcge.s16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf2]
@ CHECK: vcge.s32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf2]
@ CHECK: vcge.u8	q8, q8, q9      @ encoding: [0xf2,0x03,0x40,0xf3]
@ CHECK: vcge.u16	q8, q8, q9      @ encoding: [0xf2,0x03,0x50,0xf3]
@ CHECK: vcge.u32	q8, q8, q9      @ encoding: [0xf2,0x03,0x60,0xf3]
@ CHECK: vcge.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x40,0xf3]
@ CHECK: vacge.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x40,0xf3]
@ CHECK: vacge.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x40,0xf3]

	vcgt.s8	d16, d16, d17
	vcgt.s16	d16, d16, d17
	vcgt.s32	d16, d16, d17
	vcgt.u8	d16, d16, d17
	vcgt.u16	d16, d16, d17
	vcgt.u32	d16, d16, d17
	vcgt.f32	d16, d16, d17
	vcgt.s8	q8, q8, q9
	vcgt.s16	q8, q8, q9
	vcgt.s32	q8, q8, q9
	vcgt.u8	q8, q8, q9
	vcgt.u16	q8, q8, q9
	vcgt.u32	q8, q8, q9
	vcgt.f32	q8, q8, q9
	vacgt.f32	d16, d16, d17
	vacgt.f32	q8, q8, q9

@ CHECK: vcgt.s8	d16, d16, d17   @ encoding: [0xa1,0x03,0x40,0xf2]
@ CHECK: vcgt.s16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf2]
@ CHECK: vcgt.s32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf2]
@ CHECK: vcgt.u8	d16, d16, d17   @ encoding: [0xa1,0x03,0x40,0xf3]
@ CHECK: vcgt.u16	d16, d16, d17   @ encoding: [0xa1,0x03,0x50,0xf3]
@ CHECK: vcgt.u32	d16, d16, d17   @ encoding: [0xa1,0x03,0x60,0xf3]
@ CHECK: vcgt.f32	d16, d16, d17   @ encoding: [0xa1,0x0e,0x60,0xf3]
@ CHECK: vcgt.s8	q8, q8, q9      @ encoding: [0xe2,0x03,0x40,0xf2]
@ CHECK: vcgt.s16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf2]
@ CHECK: vcgt.s32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf2]
@ CHECK: vcgt.u8	q8, q8, q9      @ encoding: [0xe2,0x03,0x40,0xf3]
@ CHECK: vcgt.u16	q8, q8, q9      @ encoding: [0xe2,0x03,0x50,0xf3]
@ CHECK: vcgt.u32	q8, q8, q9      @ encoding: [0xe2,0x03,0x60,0xf3]
@ CHECK: vcgt.f32	q8, q8, q9      @ encoding: [0xe2,0x0e,0x60,0xf3]
@ CHECK: vacgt.f32	d16, d16, d17   @ encoding: [0xb1,0x0e,0x60,0xf3]
@ CHECK: vacgt.f32	q8, q8, q9      @ encoding: [0xf2,0x0e,0x60,0xf3]

	vtst.8	d16, d16, d17
	vtst.16	d16, d16, d17
	vtst.32	d16, d16, d17
	vtst.8	q8, q8, q9
	vtst.16	q8, q8, q9
	vtst.32	q8, q8, q9

@ CHECK: vtst.8	d16, d16, d17           @ encoding: [0xb1,0x08,0x40,0xf2]
@ CHECK: vtst.16	d16, d16, d17   @ encoding: [0xb1,0x08,0x50,0xf2]
@ CHECK: vtst.32	d16, d16, d17   @ encoding: [0xb1,0x08,0x60,0xf2]
@ CHECK: vtst.8	q8, q8, q9              @ encoding: [0xf2,0x08,0x40,0xf2]
@ CHECK: vtst.16	q8, q8, q9      @ encoding: [0xf2,0x08,0x50,0xf2]
@ CHECK: vtst.32	q8, q8, q9      @ encoding: [0xf2,0x08,0x60,0xf2]

	vceq.i8	d16, d16, #0
	vcge.s8	d16, d16, #0
	vcle.s8	d16, d16, #0
	vcgt.s8	d16, d16, #0
	vclt.s8	d16, d16, #0

@ CHECK: vceq.i8	d16, d16, #0    @ encoding: [0x20,0x01,0xf1,0xf3]
@ CHECK: vcge.s8	d16, d16, #0    @ encoding: [0xa0,0x00,0xf1,0xf3]
@ CHECK: vcle.s8	d16, d16, #0    @ encoding: [0xa0,0x01,0xf1,0xf3]
@ CHECK: vcgt.s8	d16, d16, #0    @ encoding: [0x20,0x00,0xf1,0xf3]
@ CHECK: vclt.s8	d16, d16, #0    @ encoding: [0x20,0x02,0xf1,0xf3]


        vclt.s8 q12, q13, q3
        vclt.s16 q12, q13, q3
        vclt.s32 q12, q13, q3
        vclt.u8 q12, q13, q3
        vclt.u16 q12, q13, q3
        vclt.u32 q12, q13, q3
        vclt.f32 q12, q13, q3

        vclt.s8 d12, d13, d3
        vclt.s16 d12, d13, d3
        vclt.s32 d12, d13, d3
        vclt.u8 d12, d13, d3
        vclt.u16 d12, d13, d3
        vclt.u32 d12, d13, d3
        vclt.f32 d12, d13, d3

@ CHECK: vcgt.s8	q12, q3, q13    @ encoding: [0x6a,0x83,0x46,0xf2]
@ CHECK: vcgt.s16	q12, q3, q13    @ encoding: [0x6a,0x83,0x56,0xf2]
@ CHECK: vcgt.s32	q12, q3, q13    @ encoding: [0x6a,0x83,0x66,0xf2]
@ CHECK: vcgt.u8	q12, q3, q13    @ encoding: [0x6a,0x83,0x46,0xf3]
@ CHECK: vcgt.u16	q12, q3, q13    @ encoding: [0x6a,0x83,0x56,0xf3]
@ CHECK: vcgt.u32	q12, q3, q13    @ encoding: [0x6a,0x83,0x66,0xf3]
@ CHECK: vcgt.f32	q12, q3, q13    @ encoding: [0x6a,0x8e,0x66,0xf3]

@ CHECK: vcgt.s8	d12, d3, d13    @ encoding: [0x0d,0xc3,0x03,0xf2]
@ CHECK: vcgt.s16	d12, d3, d13    @ encoding: [0x0d,0xc3,0x13,0xf2]
@ CHECK: vcgt.s32	d12, d3, d13    @ encoding: [0x0d,0xc3,0x23,0xf2]
@ CHECK: vcgt.u8	d12, d3, d13    @ encoding: [0x0d,0xc3,0x03,0xf3]
@ CHECK: vcgt.u16	d12, d3, d13    @ encoding: [0x0d,0xc3,0x13,0xf3]
@ CHECK: vcgt.u32	d12, d3, d13    @ encoding: [0x0d,0xc3,0x23,0xf3]
@ CHECK: vcgt.f32	d12, d3, d13    @ encoding: [0x0d,0xce,0x23,0xf3]

	vcle.s8	d16, d16, d17
	vcle.s16 d16, d16, d17
	vcle.s32 d16, d16, d17
	vcle.u8	d16, d16, d17
	vcle.u16 d16, d16, d17
	vcle.u32 d16, d16, d17
	vcle.f32 d16, d16, d17
	vcle.s8	q8, q8, q9
	vcle.s16 q8, q8, q9
	vcle.s32 q8, q8, q9
	vcle.u8	q8, q8, q9
	vcle.u16 q8, q8, q9
	vcle.u32 q8, q8, q9
	vcle.f32 q8, q8, q9

@ CHECK: vcge.s8	d16, d17, d16           @ encoding: [0xb0,0x03,0x41,0xf2]
@ CHECK: vcge.s16	d16, d17, d16   @ encoding: [0xb0,0x03,0x51,0xf2]
@ CHECK: vcge.s32	d16, d17, d16   @ encoding: [0xb0,0x03,0x61,0xf2]
@ CHECK: vcge.u8	d16, d17, d16           @ encoding: [0xb0,0x03,0x41,0xf3]
@ CHECK: vcge.u16	d16, d17, d16   @ encoding: [0xb0,0x03,0x51,0xf3]
@ CHECK: vcge.u32	d16, d17, d16   @ encoding: [0xb0,0x03,0x61,0xf3]
@ CHECK: vcge.f32	d16, d17, d16   @ encoding: [0xa0,0x0e,0x41,0xf3]
@ CHECK: vcge.s8	q8, q9, q8              @ encoding: [0xf0,0x03,0x42,0xf2]
@ CHECK: vcge.s16	q8, q9, q8      @ encoding: [0xf0,0x03,0x52,0xf2]
@ CHECK: vcge.s32	q8, q9, q8      @ encoding: [0xf0,0x03,0x62,0xf2]
@ CHECK: vcge.u8	q8, q9, q8              @ encoding: [0xf0,0x03,0x42,0xf3]
@ CHECK: vcge.u16	q8, q9, q8      @ encoding: [0xf0,0x03,0x52,0xf3]
@ CHECK: vcge.u32	q8, q9, q8      @ encoding: [0xf0,0x03,0x62,0xf3]
@ CHECK: vcge.f32	q8, q9, q8      @ encoding: [0xe0,0x0e,0x42,0xf3]
