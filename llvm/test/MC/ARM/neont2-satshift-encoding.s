@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vqshl.s8	d16, d16, d17   @ encoding: [0x41,0xef,0xb0,0x04]
	vqshl.s8	d16, d16, d17
@ CHECK: vqshl.s16	d16, d16, d17   @ encoding: [0x51,0xef,0xb0,0x04]
	vqshl.s16	d16, d16, d17
@ CHECK: vqshl.s32	d16, d16, d17   @ encoding: [0x61,0xef,0xb0,0x04]
	vqshl.s32	d16, d16, d17
@ CHECK: vqshl.s64	d16, d16, d17   @ encoding: [0x71,0xef,0xb0,0x04]
	vqshl.s64	d16, d16, d17
@ CHECK: vqshl.u8	d16, d16, d17   @ encoding: [0x41,0xff,0xb0,0x04]
	vqshl.u8	d16, d16, d17
@ CHECK: vqshl.u16	d16, d16, d17   @ encoding: [0x51,0xff,0xb0,0x04]
	vqshl.u16	d16, d16, d17
@ CHECK: vqshl.u32	d16, d16, d17   @ encoding: [0x61,0xff,0xb0,0x04]
	vqshl.u32	d16, d16, d17
@ CHECK: vqshl.u64	d16, d16, d17   @ encoding: [0x71,0xff,0xb0,0x04]
	vqshl.u64	d16, d16, d17
@ CHECK: vqshl.s8	q8, q8, q9      @ encoding: [0x42,0xef,0xf0,0x04]
	vqshl.s8	q8, q8, q9
@ CHECK: vqshl.s16	q8, q8, q9      @ encoding: [0x52,0xef,0xf0,0x04]
	vqshl.s16	q8, q8, q9
@ CHECK: vqshl.s32	q8, q8, q9      @ encoding: [0x62,0xef,0xf0,0x04]
	vqshl.s32	q8, q8, q9
@ CHECK: vqshl.s64	q8, q8, q9      @ encoding: [0x72,0xef,0xf0,0x04]
	vqshl.s64	q8, q8, q9
@ CHECK: vqshl.u8	q8, q8, q9      @ encoding: [0x42,0xff,0xf0,0x04]
	vqshl.u8	q8, q8, q9
@ CHECK: vqshl.u16	q8, q8, q9      @ encoding: [0x52,0xff,0xf0,0x04]
	vqshl.u16	q8, q8, q9
@ CHECK: vqshl.u32	q8, q8, q9      @ encoding: [0x62,0xff,0xf0,0x04]
	vqshl.u32	q8, q8, q9
@ CHECK: vqshl.u64	q8, q8, q9      @ encoding: [0x72,0xff,0xf0,0x04]
	vqshl.u64	q8, q8, q9
@ CHECK: vqshl.s8	d16, d16, #7    @ encoding: [0xcf,0xef,0x30,0x07]
	vqshl.s8	d16, d16, #7
@ CHECK: vqshl.s16	d16, d16, #15   @ encoding: [0xdf,0xef,0x30,0x07]
	vqshl.s16	d16, d16, #15
@ CHECK: vqshl.s32	d16, d16, #31   @ encoding: [0xff,0xef,0x30,0x07]
	vqshl.s32	d16, d16, #31
@ CHECK: vqshl.s64	d16, d16, #63   @ encoding: [0xff,0xef,0xb0,0x07]
	vqshl.s64	d16, d16, #63
@ CHECK: vqshl.u8	d16, d16, #7    @ encoding: [0xcf,0xff,0x30,0x07]
	vqshl.u8	d16, d16, #7
@ CHECK: vqshl.u16	d16, d16, #15   @ encoding: [0xdf,0xff,0x30,0x07]
	vqshl.u16	d16, d16, #15
@ CHECK: vqshl.u32	d16, d16, #31   @ encoding: [0xff,0xff,0x30,0x07]
	vqshl.u32	d16, d16, #31
@ CHECK: vqshl.u64	d16, d16, #63   @ encoding: [0xff,0xff,0xb0,0x07]
	vqshl.u64	d16, d16, #63
@ CHECK: vqshlu.s8	d16, d16, #7    @ encoding: [0xcf,0xff,0x30,0x06]
	vqshlu.s8	d16, d16, #7
@ CHECK: vqshlu.s16	d16, d16, #15   @ encoding: [0xdf,0xff,0x30,0x06]
	vqshlu.s16	d16, d16, #15
@ CHECK: vqshlu.s32	d16, d16, #31   @ encoding: [0xff,0xff,0x30,0x06]
	vqshlu.s32	d16, d16, #31
@ CHECK: vqshlu.s64	d16, d16, #63   @ encoding: [0xff,0xff,0xb0,0x06]
	vqshlu.s64	d16, d16, #63
@ CHECK: vqshl.s8	q8, q8, #7      @ encoding: [0xcf,0xef,0x70,0x07]
	vqshl.s8	q8, q8, #7
@ CHECK: vqshl.s16	q8, q8, #15     @ encoding: [0xdf,0xef,0x70,0x07]
	vqshl.s16	q8, q8, #15
@ CHECK: vqshl.s32	q8, q8, #31     @ encoding: [0xff,0xef,0x70,0x07]
	vqshl.s32	q8, q8, #31
@ CHECK: vqshl.s64	q8, q8, #63     @ encoding: [0xff,0xef,0xf0,0x07]
	vqshl.s64	q8, q8, #63
@ CHECK: vqshl.u8	q8, q8, #7      @ encoding: [0xcf,0xff,0x70,0x07]
	vqshl.u8	q8, q8, #7
@ CHECK: vqshl.u16	q8, q8, #15     @ encoding: [0xdf,0xff,0x70,0x07]
	vqshl.u16	q8, q8, #15
@ CHECK: vqshl.u32	q8, q8, #31     @ encoding: [0xff,0xff,0x70,0x07]
	vqshl.u32	q8, q8, #31
@ CHECK: vqshl.u64	q8, q8, #63     @ encoding: [0xff,0xff,0xf0,0x07]
	vqshl.u64	q8, q8, #63
@ CHECK: vqshlu.s8	q8, q8, #7      @ encoding: [0xcf,0xff,0x70,0x06]
	vqshlu.s8	q8, q8, #7
@ CHECK: vqshlu.s16	q8, q8, #15     @ encoding: [0xdf,0xff,0x70,0x06]
	vqshlu.s16	q8, q8, #15
@ CHECK: vqshlu.s32	q8, q8, #31     @ encoding: [0xff,0xff,0x70,0x06]
	vqshlu.s32	q8, q8, #31
@ CHECK: vqshlu.s64	q8, q8, #63     @ encoding: [0xff,0xff,0xf0,0x06]
	vqshlu.s64	q8, q8, #63
@ CHECK:   vqrshl.s8	d16, d16, d17   @ encoding: [0x41,0xef,0xb0,0x05]
	vqrshl.s8	d16, d16, d17
@ CHECK: vqrshl.s16	d16, d16, d17   @ encoding: [0x51,0xef,0xb0,0x05]
	vqrshl.s16	d16, d16, d17
@ CHECK: vqrshl.s32	d16, d16, d17   @ encoding: [0x61,0xef,0xb0,0x05]
	vqrshl.s32	d16, d16, d17
@ CHECK: vqrshl.s64	d16, d16, d17   @ encoding: [0x71,0xef,0xb0,0x05]
	vqrshl.s64	d16, d16, d17
@ CHECK: vqrshl.u8	d16, d16, d17   @ encoding: [0x41,0xff,0xb0,0x05]
	vqrshl.u8	d16, d16, d17
@ CHECK: vqrshl.u16	d16, d16, d17   @ encoding: [0x51,0xff,0xb0,0x05]
	vqrshl.u16	d16, d16, d17
@ CHECK: vqrshl.u32	d16, d16, d17   @ encoding: [0x61,0xff,0xb0,0x05]
	vqrshl.u32	d16, d16, d17
@ CHECK: vqrshl.u64	d16, d16, d17   @ encoding: [0x71,0xff,0xb0,0x05]
	vqrshl.u64	d16, d16, d17
@ CHECK: vqrshl.s8	q8, q8, q9      @ encoding: [0x42,0xef,0xf0,0x05]
	vqrshl.s8	q8, q8, q9
@ CHECK: vqrshl.s16	q8, q8, q9      @ encoding: [0x52,0xef,0xf0,0x05]
	vqrshl.s16	q8, q8, q9
@ CHECK: vqrshl.s32	q8, q8, q9      @ encoding: [0x62,0xef,0xf0,0x05]
	vqrshl.s32	q8, q8, q9
@ CHECK: vqrshl.s64	q8, q8, q9      @ encoding: [0x72,0xef,0xf0,0x05]
	vqrshl.s64	q8, q8, q9
@ CHECK: vqrshl.u8	q8, q8, q9      @ encoding: [0x42,0xff,0xf0,0x05]
	vqrshl.u8	q8, q8, q9
@ CHECK: vqrshl.u16	q8, q8, q9      @ encoding: [0x52,0xff,0xf0,0x05]
	vqrshl.u16	q8, q8, q9
@ CHECK: vqrshl.u32	q8, q8, q9      @ encoding: [0x62,0xff,0xf0,0x05]
	vqrshl.u32	q8, q8, q9
@ CHECK: vqrshl.u64	q8, q8, q9      @ encoding: [0x72,0xff,0xf0,0x05]
	vqrshl.u64	q8, q8, q9
@ CHECK: vqshrn.s16	d16, q8, #8     @ encoding: [0xc8,0xef,0x30,0x09]
	vqshrn.s16	d16, q8, #8
@ CHECK: vqshrn.s32	d16, q8, #16    @ encoding: [0xd0,0xef,0x30,0x09]
	vqshrn.s32	d16, q8, #16
@ CHECK: vqshrn.s64	d16, q8, #32    @ encoding: [0xe0,0xef,0x30,0x09]
	vqshrn.s64	d16, q8, #32
@ CHECK: vqshrn.u16	d16, q8, #8     @ encoding: [0xc8,0xff,0x30,0x09]
	vqshrn.u16	d16, q8, #8
@ CHECK: vqshrn.u32	d16, q8, #16    @ encoding: [0xd0,0xff,0x30,0x09]
	vqshrn.u32	d16, q8, #16
@ CHECK: vqshrn.u64	d16, q8, #32    @ encoding: [0xe0,0xff,0x30,0x09]
	vqshrn.u64	d16, q8, #32
@ CHECK: vqshrun.s16	d16, q8, #8     @ encoding: [0xc8,0xff,0x30,0x08]
	vqshrun.s16	d16, q8, #8
@ CHECK: vqshrun.s32	d16, q8, #16    @ encoding: [0xd0,0xff,0x30,0x08]
	vqshrun.s32	d16, q8, #16
@ CHECK: vqshrun.s64	d16, q8, #32    @ encoding: [0xe0,0xff,0x30,0x08]
	vqshrun.s64	d16, q8, #32
@ CHECK: vqrshrn.s16	d16, q8, #8     @ encoding: [0xc8,0xef,0x70,0x09]
	vqrshrn.s16	d16, q8, #8
@ CHECK: vqrshrn.s32	d16, q8, #16    @ encoding: [0xd0,0xef,0x70,0x09]
	vqrshrn.s32	d16, q8, #16
@ CHECK: vqrshrn.s64	d16, q8, #32    @ encoding: [0xe0,0xef,0x70,0x09]
	vqrshrn.s64	d16, q8, #32
@ CHECK: vqrshrn.u16	d16, q8, #8     @ encoding: [0xc8,0xff,0x70,0x09]
	vqrshrn.u16	d16, q8, #8
@ CHECK: vqrshrn.u32	d16, q8, #16    @ encoding: [0xd0,0xff,0x70,0x09]
	vqrshrn.u32	d16, q8, #16
@ CHECK: vqrshrn.u64	d16, q8, #32    @ encoding: [0xe0,0xff,0x70,0x09]
	vqrshrn.u64	d16, q8, #32
@ CHECK: vqrshrun.s16	d16, q8, #8     @ encoding: [0xc8,0xff,0x70,0x08]
	vqrshrun.s16	d16, q8, #8
@ CHECK: vqrshrun.s32	d16, q8, #16    @ encoding: [0xd0,0xff,0x70,0x08]
	vqrshrun.s32	d16, q8, #16
@ CHECK: vqrshrun.s64	d16, q8, #32    @ encoding: [0xe0,0xff,0x70,0x08]
	vqrshrun.s64	d16, q8, #32
