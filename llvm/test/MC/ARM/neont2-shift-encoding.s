@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vshl.u8	d16, d17, d16           @ encoding: [0x40,0xff,0xa1,0x04]
	vshl.u8	d16, d17, d16
@ CHECK: vshl.u16	d16, d17, d16   @ encoding: [0x50,0xff,0xa1,0x04]
	vshl.u16	d16, d17, d16
@ CHECK: vshl.u32	d16, d17, d16   @ encoding: [0x60,0xff,0xa1,0x04]
	vshl.u32	d16, d17, d16
@ CHECK: vshl.u64	d16, d17, d16   @ encoding: [0x70,0xff,0xa1,0x04]
	vshl.u64	d16, d17, d16
@ CHECK: vshl.i8	d16, d16, #7            @ encoding: [0xcf,0xef,0x30,0x05]
	vshl.i8	d16, d16, #7
@ CHECK: vshl.i16	d16, d16, #15   @ encoding: [0xdf,0xef,0x30,0x05]
	vshl.i16	d16, d16, #15
@ CHECK: vshl.i32	d16, d16, #31   @ encoding: [0xff,0xef,0x30,0x05]
	vshl.i32	d16, d16, #31
@ CHECK: vshl.i64	d16, d16, #63   @ encoding: [0xff,0xef,0xb0,0x05]
	vshl.i64	d16, d16, #63
@ CHECK: vshl.u8	q8, q9, q8              @ encoding: [0x40,0xff,0xe2,0x04]
	vshl.u8	q8, q9, q8
@ CHECK: vshl.u16	q8, q9, q8      @ encoding: [0x50,0xff,0xe2,0x04]
	vshl.u16	q8, q9, q8
@ CHECK: vshl.u32	q8, q9, q8      @ encoding: [0x60,0xff,0xe2,0x04]
	vshl.u32	q8, q9, q8
@ CHECK: vshl.u64	q8, q9, q8      @ encoding: [0x70,0xff,0xe2,0x04]
	vshl.u64	q8, q9, q8
@ CHECK: vshl.i8	q8, q8, #7              @ encoding: [0xcf,0xef,0x70,0x05]
	vshl.i8	q8, q8, #7
@ CHECK: vshl.i16	q8, q8, #15     @ encoding: [0xdf,0xef,0x70,0x05]
	vshl.i16	q8, q8, #15
@ CHECK: vshl.i32	q8, q8, #31     @ encoding: [0xff,0xef,0x70,0x05]
	vshl.i32	q8, q8, #31
@ CHECK: vshl.i64	q8, q8, #63     @ encoding: [0xff,0xef,0xf0,0x05]
	vshl.i64	q8, q8, #63
@ CHECK: vshr.u8	d16, d16, #8            @ encoding: [0xc8,0xff,0x30,0x00]
	vshr.u8	d16, d16, #8
@ CHECK: vshr.u16	d16, d16, #16   @ encoding: [0xd0,0xff,0x30,0x00]
	vshr.u16	d16, d16, #16
@ CHECK: vshr.u32	d16, d16, #32   @ encoding: [0xe0,0xff,0x30,0x00]
	vshr.u32	d16, d16, #32
@ CHECK: vshr.u64	d16, d16, #64   @ encoding: [0xc0,0xff,0xb0,0x00]
	vshr.u64	d16, d16, #64
@ CHECK: vshr.u8	q8, q8, #8              @ encoding: [0xc8,0xff,0x70,0x00]
	vshr.u8	q8, q8, #8
@ CHECK: vshr.u16	q8, q8, #16     @ encoding: [0xd0,0xff,0x70,0x00]
	vshr.u16	q8, q8, #16
@ CHECK: vshr.u32	q8, q8, #32     @ encoding: [0xe0,0xff,0x70,0x00]
	vshr.u32	q8, q8, #32
@ CHECK: vshr.u64	q8, q8, #64     @ encoding: [0xc0,0xff,0xf0,0x00]
	vshr.u64	q8, q8, #64
@ CHECK: vshr.s8	d16, d16, #8            @ encoding: [0xc8,0xef,0x30,0x00]
	vshr.s8	d16, d16, #8
@ CHECK: vshr.s16	d16, d16, #16   @ encoding: [0xd0,0xef,0x30,0x00]
	vshr.s16	d16, d16, #16
@ CHECK: vshr.s32	d16, d16, #32   @ encoding: [0xe0,0xef,0x30,0x00]
	vshr.s32	d16, d16, #32
@ CHECK: vshr.s64	d16, d16, #64   @ encoding: [0xc0,0xef,0xb0,0x00]
	vshr.s64	d16, d16, #64
@ CHECK: vshr.s8	q8, q8, #8              @ encoding: [0xc8,0xef,0x70,0x00]
	vshr.s8	q8, q8, #8
@ CHECK: vshr.s16	q8, q8, #16     @ encoding: [0xd0,0xef,0x70,0x00]
	vshr.s16	q8, q8, #16
@ CHECK: vshr.s32	q8, q8, #32     @ encoding: [0xe0,0xef,0x70,0x00]
	vshr.s32	q8, q8, #32
@ CHECK: vshr.s64	q8, q8, #64     @ encoding: [0xc0,0xef,0xf0,0x00]
	vshr.s64	q8, q8, #64
@ CHECK: vshll.s8	q8, d16, #7     @ encoding: [0xcf,0xef,0x30,0x0a]
	vshll.s8	q8, d16, #7
@ CHECK: vshll.s16	q8, d16, #15    @ encoding: [0xdf,0xef,0x30,0x0a]
	vshll.s16	q8, d16, #15
@ CHECK: vshll.s32	q8, d16, #31    @ encoding: [0xff,0xef,0x30,0x0a]
	vshll.s32	q8, d16, #31
@ CHECK: vshll.u8	q8, d16, #7     @ encoding: [0xcf,0xff,0x30,0x0a]
	vshll.u8	q8, d16, #7
@ CHECK: vshll.u16	q8, d16, #15    @ encoding: [0xdf,0xff,0x30,0x0a]
	vshll.u16	q8, d16, #15
@ CHECK: vshll.u32	q8, d16, #31    @ encoding: [0xff,0xff,0x30,0x0a]
	vshll.u32	q8, d16, #31
@ CHECK: vshll.i8	q8, d16, #8     @ encoding: [0xf2,0xff,0x20,0x03]
	vshll.i8	q8, d16, #8
@ CHECK: vshll.i16	q8, d16, #16    @ encoding: [0xf6,0xff,0x20,0x03]
	vshll.i16	q8, d16, #16
@ CHECK: vshll.i32	q8, d16, #32    @ encoding: [0xfa,0xff,0x20,0x03]
	vshll.i32	q8, d16, #32
@ CHECK: vshrn.i16	d16, q8, #8     @ encoding: [0xc8,0xef,0x30,0x08]
	vshrn.i16	d16, q8, #8
@ CHECK: vshrn.i32	d16, q8, #16    @ encoding: [0xd0,0xef,0x30,0x08]
	vshrn.i32	d16, q8, #16
@ CHECK: vshrn.i64	d16, q8, #32    @ encoding: [0xe0,0xef,0x30,0x08]
	vshrn.i64	d16, q8, #32
@ CHECK: vrshl.s8	d16, d17, d16   @ encoding: [0x40,0xef,0xa1,0x05]
	vrshl.s8	d16, d17, d16
@ CHECK: vrshl.s16	d16, d17, d16   @ encoding: [0x50,0xef,0xa1,0x05]
	vrshl.s16	d16, d17, d16
@ CHECK: vrshl.s32	d16, d17, d16   @ encoding: [0x60,0xef,0xa1,0x05]
	vrshl.s32	d16, d17, d16
@ CHECK: vrshl.s64	d16, d17, d16   @ encoding: [0x70,0xef,0xa1,0x05]
	vrshl.s64	d16, d17, d16
@ CHECK: vrshl.u8	d16, d17, d16   @ encoding: [0x40,0xff,0xa1,0x05]
	vrshl.u8	d16, d17, d16
@ CHECK: vrshl.u16	d16, d17, d16   @ encoding: [0x50,0xff,0xa1,0x05]
	vrshl.u16	d16, d17, d16
@ CHECK: vrshl.u32	d16, d17, d16   @ encoding: [0x60,0xff,0xa1,0x05]
	vrshl.u32	d16, d17, d16
@ CHECK: vrshl.u64	d16, d17, d16   @ encoding: [0x70,0xff,0xa1,0x05]
	vrshl.u64	d16, d17, d16
@ CHECK: vrshl.s8	q8, q9, q8      @ encoding: [0x40,0xef,0xe2,0x05]
	vrshl.s8	q8, q9, q8
@ CHECK: vrshl.s16	q8, q9, q8      @ encoding: [0x50,0xef,0xe2,0x05]
	vrshl.s16	q8, q9, q8
@ CHECK: vrshl.s32	q8, q9, q8      @ encoding: [0x60,0xef,0xe2,0x05]
	vrshl.s32	q8, q9, q8
@ CHECK: vrshl.s64	q8, q9, q8      @ encoding: [0x70,0xef,0xe2,0x05]
	vrshl.s64	q8, q9, q8
@ CHECK: vrshl.u8	q8, q9, q8      @ encoding: [0x40,0xff,0xe2,0x05]
	vrshl.u8	q8, q9, q8
@ CHECK: vrshl.u16	q8, q9, q8      @ encoding: [0x50,0xff,0xe2,0x05]
	vrshl.u16	q8, q9, q8
@ CHECK: vrshl.u32	q8, q9, q8      @ encoding: [0x60,0xff,0xe2,0x05]
	vrshl.u32	q8, q9, q8
@ CHECK: vrshl.u64	q8, q9, q8      @ encoding: [0x70,0xff,0xe2,0x05]
	vrshl.u64	q8, q9, q8
@ CHECK: vrshr.s8	d16, d16, #8    @ encoding: [0xc8,0xef,0x30,0x02]
	vrshr.s8	d16, d16, #8
@ CHECK: vrshr.s16	d16, d16, #16   @ encoding: [0xd0,0xef,0x30,0x02]
	vrshr.s16	d16, d16, #16
@ CHECK: vrshr.s32	d16, d16, #32   @ encoding: [0xe0,0xef,0x30,0x02]
	vrshr.s32	d16, d16, #32
@ CHECK: vrshr.s64	d16, d16, #64   @ encoding: [0xc0,0xef,0xb0,0x02]
	vrshr.s64	d16, d16, #64
@ CHECK: vrshr.u8	d16, d16, #8    @ encoding: [0xc8,0xff,0x30,0x02]
	vrshr.u8	d16, d16, #8
@ CHECK: vrshr.u16	d16, d16, #16   @ encoding: [0xd0,0xff,0x30,0x02]
	vrshr.u16	d16, d16, #16
@ CHECK: vrshr.u32	d16, d16, #32   @ encoding: [0xe0,0xff,0x30,0x02]
	vrshr.u32	d16, d16, #32
@ CHECK: vrshr.u64	d16, d16, #64   @ encoding: [0xc0,0xff,0xb0,0x02]
	vrshr.u64	d16, d16, #64
@ CHECK: vrshr.s8	q8, q8, #8      @ encoding: [0xc8,0xef,0x70,0x02]
	vrshr.s8	q8, q8, #8
@ CHECK: vrshr.s16	q8, q8, #16     @ encoding: [0xd0,0xef,0x70,0x02]
	vrshr.s16	q8, q8, #16
@ CHECK: vrshr.s32	q8, q8, #32     @ encoding: [0xe0,0xef,0x70,0x02]
	vrshr.s32	q8, q8, #32
@ CHECK: vrshr.s64	q8, q8, #64     @ encoding: [0xc0,0xef,0xf0,0x02]
	vrshr.s64	q8, q8, #64
@ CHECK: vrshr.u8	q8, q8, #8      @ encoding: [0xc8,0xff,0x70,0x02]
	vrshr.u8	q8, q8, #8
@ CHECK: vrshr.u16	q8, q8, #16     @ encoding: [0xd0,0xff,0x70,0x02]
	vrshr.u16	q8, q8, #16
@ CHECK: vrshr.u32	q8, q8, #32     @ encoding: [0xe0,0xff,0x70,0x02]
	vrshr.u32	q8, q8, #32
@ CHECK: vrshr.u64	q8, q8, #64     @ encoding: [0xc0,0xff,0xf0,0x02]
	vrshr.u64	q8, q8, #64
@ CHECK: vrshrn.i16	d16, q8, #8     @ encoding: [0xc8,0xef,0x70,0x08]
	vrshrn.i16	d16, q8, #8
@ CHECK: vrshrn.i32	d16, q8, #16    @ encoding: [0xd0,0xef,0x70,0x08]
	vrshrn.i32	d16, q8, #16
@ CHECK: vrshrn.i64	d16, q8, #32    @ encoding: [0xe0,0xef,0x70,0x08]
	vrshrn.i64	d16, q8, #32
