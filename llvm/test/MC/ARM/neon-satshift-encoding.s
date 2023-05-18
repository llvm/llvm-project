@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vqshl.s8	d16, d16, d17   @ encoding: [0xb0,0x04,0x41,0xf2]
	vqshl.s8	d16, d16, d17
@ CHECK: vqshl.s16	d16, d16, d17   @ encoding: [0xb0,0x04,0x51,0xf2]
	vqshl.s16	d16, d16, d17
@ CHECK: vqshl.s32	d16, d16, d17   @ encoding: [0xb0,0x04,0x61,0xf2]
	vqshl.s32	d16, d16, d17
@ CHECK: vqshl.s64	d16, d16, d17   @ encoding: [0xb0,0x04,0x71,0xf2]
	vqshl.s64	d16, d16, d17
@ CHECK: vqshl.u8	d16, d16, d17   @ encoding: [0xb0,0x04,0x41,0xf3]
	vqshl.u8	d16, d16, d17
@ CHECK: vqshl.u16	d16, d16, d17   @ encoding: [0xb0,0x04,0x51,0xf3]
	vqshl.u16	d16, d16, d17
@ CHECK: vqshl.u32	d16, d16, d17   @ encoding: [0xb0,0x04,0x61,0xf3]
	vqshl.u32	d16, d16, d17
@ CHECK: vqshl.u64	d16, d16, d17   @ encoding: [0xb0,0x04,0x71,0xf3]
	vqshl.u64	d16, d16, d17
@ CHECK: vqshl.s8	q8, q8, q9      @ encoding: [0xf0,0x04,0x42,0xf2]
	vqshl.s8	q8, q8, q9
@ CHECK: vqshl.s16	q8, q8, q9      @ encoding: [0xf0,0x04,0x52,0xf2]
	vqshl.s16	q8, q8, q9
@ CHECK: vqshl.s32	q8, q8, q9      @ encoding: [0xf0,0x04,0x62,0xf2]
	vqshl.s32	q8, q8, q9
@ CHECK: vqshl.s64	q8, q8, q9      @ encoding: [0xf0,0x04,0x72,0xf2]
	vqshl.s64	q8, q8, q9
@ CHECK: vqshl.u8	q8, q8, q9      @ encoding: [0xf0,0x04,0x42,0xf3]
	vqshl.u8	q8, q8, q9
@ CHECK: vqshl.u16	q8, q8, q9      @ encoding: [0xf0,0x04,0x52,0xf3]
	vqshl.u16	q8, q8, q9
@ CHECK: vqshl.u32	q8, q8, q9      @ encoding: [0xf0,0x04,0x62,0xf3]
	vqshl.u32	q8, q8, q9
@ CHECK: vqshl.u64	q8, q8, q9      @ encoding: [0xf0,0x04,0x72,0xf3]
	vqshl.u64	q8, q8, q9
@ CHECK: vqshl.s8	d16, d16, #7    @ encoding: [0x30,0x07,0xcf,0xf2]
	vqshl.s8	d16, d16, #7
@ CHECK: vqshl.s16	d16, d16, #15   @ encoding: [0x30,0x07,0xdf,0xf2]
	vqshl.s16	d16, d16, #15
@ CHECK: vqshl.s32	d16, d16, #31   @ encoding: [0x30,0x07,0xff,0xf2]
	vqshl.s32	d16, d16, #31
@ CHECK: vqshl.s64	d16, d16, #63   @ encoding: [0xb0,0x07,0xff,0xf2]
	vqshl.s64	d16, d16, #63
@ CHECK: vqshl.u8	d16, d16, #7    @ encoding: [0x30,0x07,0xcf,0xf3]
	vqshl.u8	d16, d16, #7
@ CHECK: vqshl.u16	d16, d16, #15   @ encoding: [0x30,0x07,0xdf,0xf3]
	vqshl.u16	d16, d16, #15
@ CHECK: vqshl.u32	d16, d16, #31   @ encoding: [0x30,0x07,0xff,0xf3]
	vqshl.u32	d16, d16, #31
@ CHECK: vqshl.u64	d16, d16, #63   @ encoding: [0xb0,0x07,0xff,0xf3]
	vqshl.u64	d16, d16, #63
@ CHECK: vqshlu.s8	d16, d16, #7    @ encoding: [0x30,0x06,0xcf,0xf3]
	vqshlu.s8	d16, d16, #7
@ CHECK: vqshlu.s16	d16, d16, #15   @ encoding: [0x30,0x06,0xdf,0xf3]
	vqshlu.s16	d16, d16, #15
@ CHECK: vqshlu.s32	d16, d16, #31   @ encoding: [0x30,0x06,0xff,0xf3]
	vqshlu.s32	d16, d16, #31
@ CHECK: vqshlu.s64	d16, d16, #63   @ encoding: [0xb0,0x06,0xff,0xf3]
	vqshlu.s64	d16, d16, #63
@ CHECK: vqshl.s8	q8, q8, #7      @ encoding: [0x70,0x07,0xcf,0xf2]
	vqshl.s8	q8, q8, #7
@ CHECK: vqshl.s16	q8, q8, #15     @ encoding: [0x70,0x07,0xdf,0xf2]
	vqshl.s16	q8, q8, #15
@ CHECK: vqshl.s32	q8, q8, #31     @ encoding: [0x70,0x07,0xff,0xf2]
	vqshl.s32	q8, q8, #31
@ CHECK: vqshl.s64	q8, q8, #63     @ encoding: [0xf0,0x07,0xff,0xf2]
	vqshl.s64	q8, q8, #63
@ CHECK: vqshl.u8	q8, q8, #7      @ encoding: [0x70,0x07,0xcf,0xf3]
	vqshl.u8	q8, q8, #7
@ CHECK: vqshl.u16	q8, q8, #15     @ encoding: [0x70,0x07,0xdf,0xf3]
	vqshl.u16	q8, q8, #15
@ CHECK: vqshl.u32	q8, q8, #31     @ encoding: [0x70,0x07,0xff,0xf3]
	vqshl.u32	q8, q8, #31
@ CHECK: vqshl.u64	q8, q8, #63     @ encoding: [0xf0,0x07,0xff,0xf3]
	vqshl.u64	q8, q8, #63
@ CHECK: vqshlu.s8	q8, q8, #7      @ encoding: [0x70,0x06,0xcf,0xf3]
	vqshlu.s8	q8, q8, #7
@ CHECK: vqshlu.s16	q8, q8, #15     @ encoding: [0x70,0x06,0xdf,0xf3]
	vqshlu.s16	q8, q8, #15
@ CHECK: vqshlu.s32	q8, q8, #31     @ encoding: [0x70,0x06,0xff,0xf3]
	vqshlu.s32	q8, q8, #31
@ CHECK: vqshlu.s64	q8, q8, #63     @ encoding: [0xf0,0x06,0xff,0xf3]
	vqshlu.s64	q8, q8, #63
@ CHECK:   vqrshl.s8	d16, d16, d17   @ encoding: [0xb0,0x05,0x41,0xf2]
	vqrshl.s8	d16, d16, d17
@ CHECK: vqrshl.s16	d16, d16, d17   @ encoding: [0xb0,0x05,0x51,0xf2]
	vqrshl.s16	d16, d16, d17
@ CHECK: vqrshl.s32	d16, d16, d17   @ encoding: [0xb0,0x05,0x61,0xf2]
	vqrshl.s32	d16, d16, d17
@ CHECK: vqrshl.s64	d16, d16, d17   @ encoding: [0xb0,0x05,0x71,0xf2]
	vqrshl.s64	d16, d16, d17
@ CHECK: vqrshl.u8	d16, d16, d17   @ encoding: [0xb0,0x05,0x41,0xf3]
	vqrshl.u8	d16, d16, d17
@ CHECK: vqrshl.u16	d16, d16, d17   @ encoding: [0xb0,0x05,0x51,0xf3]
	vqrshl.u16	d16, d16, d17
@ CHECK: vqrshl.u32	d16, d16, d17   @ encoding: [0xb0,0x05,0x61,0xf3]
	vqrshl.u32	d16, d16, d17
@ CHECK: vqrshl.u64	d16, d16, d17   @ encoding: [0xb0,0x05,0x71,0xf3]
	vqrshl.u64	d16, d16, d17
@ CHECK: vqrshl.s8	q8, q8, q9      @ encoding: [0xf0,0x05,0x42,0xf2]
	vqrshl.s8	q8, q8, q9
@ CHECK: vqrshl.s16	q8, q8, q9      @ encoding: [0xf0,0x05,0x52,0xf2]
	vqrshl.s16	q8, q8, q9
@ CHECK: vqrshl.s32	q8, q8, q9      @ encoding: [0xf0,0x05,0x62,0xf2]
	vqrshl.s32	q8, q8, q9
@ CHECK: vqrshl.s64	q8, q8, q9      @ encoding: [0xf0,0x05,0x72,0xf2]
	vqrshl.s64	q8, q8, q9
@ CHECK: vqrshl.u8	q8, q8, q9      @ encoding: [0xf0,0x05,0x42,0xf3]
	vqrshl.u8	q8, q8, q9
@ CHECK: vqrshl.u16	q8, q8, q9      @ encoding: [0xf0,0x05,0x52,0xf3]
	vqrshl.u16	q8, q8, q9
@ CHECK: vqrshl.u32	q8, q8, q9      @ encoding: [0xf0,0x05,0x62,0xf3]
	vqrshl.u32	q8, q8, q9
@ CHECK: vqrshl.u64	q8, q8, q9      @ encoding: [0xf0,0x05,0x72,0xf3]
	vqrshl.u64	q8, q8, q9
@ CHECK: vqshrn.s16	d16, q8, #8     @ encoding: [0x30,0x09,0xc8,0xf2]
	vqshrn.s16	d16, q8, #8
@ CHECK: vqshrn.s32	d16, q8, #16    @ encoding: [0x30,0x09,0xd0,0xf2]
	vqshrn.s32	d16, q8, #16
@ CHECK: vqshrn.s64	d16, q8, #32    @ encoding: [0x30,0x09,0xe0,0xf2]
	vqshrn.s64	d16, q8, #32
@ CHECK: vqshrn.u16	d16, q8, #8     @ encoding: [0x30,0x09,0xc8,0xf3]
	vqshrn.u16	d16, q8, #8
@ CHECK: vqshrn.u32	d16, q8, #16    @ encoding: [0x30,0x09,0xd0,0xf3]
	vqshrn.u32	d16, q8, #16
@ CHECK: vqshrn.u64	d16, q8, #32    @ encoding: [0x30,0x09,0xe0,0xf3]
	vqshrn.u64	d16, q8, #32
@ CHECK: vqshrun.s16	d16, q8, #8     @ encoding: [0x30,0x08,0xc8,0xf3]
	vqshrun.s16	d16, q8, #8
@ CHECK: vqshrun.s32	d16, q8, #16    @ encoding: [0x30,0x08,0xd0,0xf3]
	vqshrun.s32	d16, q8, #16
@ CHECK: vqshrun.s64	d16, q8, #32    @ encoding: [0x30,0x08,0xe0,0xf3]
	vqshrun.s64	d16, q8, #32
@ CHECK: vqrshrn.s16	d16, q8, #8     @ encoding: [0x70,0x09,0xc8,0xf2]
	vqrshrn.s16	d16, q8, #8
@ CHECK: vqrshrn.s32	d16, q8, #16    @ encoding: [0x70,0x09,0xd0,0xf2]
	vqrshrn.s32	d16, q8, #16
@ CHECK: vqrshrn.s64	d16, q8, #32    @ encoding: [0x70,0x09,0xe0,0xf2]
	vqrshrn.s64	d16, q8, #32
@ CHECK: vqrshrn.u16	d16, q8, #8     @ encoding: [0x70,0x09,0xc8,0xf3]
	vqrshrn.u16	d16, q8, #8
@ CHECK: vqrshrn.u32	d16, q8, #16    @ encoding: [0x70,0x09,0xd0,0xf3]
	vqrshrn.u32	d16, q8, #16
@ CHECK: vqrshrn.u64	d16, q8, #32    @ encoding: [0x70,0x09,0xe0,0xf3]
	vqrshrn.u64	d16, q8, #32
@ CHECK: vqrshrun.s16	d16, q8, #8     @ encoding: [0x70,0x08,0xc8,0xf3]
	vqrshrun.s16	d16, q8, #8
@ CHECK: vqrshrun.s32	d16, q8, #16    @ encoding: [0x70,0x08,0xd0,0xf3]
	vqrshrun.s32	d16, q8, #16
@ CHECK: vqrshrun.s64	d16, q8, #32    @ encoding: [0x70,0x08,0xe0,0xf3]
	vqrshrun.s64	d16, q8, #32
