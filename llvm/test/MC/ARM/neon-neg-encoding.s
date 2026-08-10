@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vneg.s8	d16, d16                @ encoding: [0xa0,0x03,0xf1,0xf3]
	vneg.s8	d16, d16
@ CHECK: vneg.s16	d16, d16        @ encoding: [0xa0,0x03,0xf5,0xf3]
	vneg.s16	d16, d16
@ CHECK: vneg.s32	d16, d16        @ encoding: [0xa0,0x03,0xf9,0xf3]
	vneg.s32	d16, d16
@ CHECK: vneg.f32	d16, d16        @ encoding: [0xa0,0x07,0xf9,0xf3]
	vneg.f32	d16, d16
@ CHECK: vneg.s8	q8, q8                  @ encoding: [0xe0,0x03,0xf1,0xf3]
	vneg.s8	q8, q8
@ CHECK: vneg.s16	q8, q8          @ encoding: [0xe0,0x03,0xf5,0xf3]
	vneg.s16	q8, q8
@ CHECK: vneg.s32	q8, q8          @ encoding: [0xe0,0x03,0xf9,0xf3]
	vneg.s32	q8, q8
@ CHECK: vneg.f32	q8, q8          @ encoding: [0xe0,0x07,0xf9,0xf3]
	vneg.f32	q8, q8
@ CHECK: vqneg.s8	d16, d16        @ encoding: [0xa0,0x07,0xf0,0xf3]
	vqneg.s8	d16, d16
@ CHECK: vqneg.s16	d16, d16        @ encoding: [0xa0,0x07,0xf4,0xf3]
	vqneg.s16	d16, d16
@ CHECK: vqneg.s32	d16, d16        @ encoding: [0xa0,0x07,0xf8,0xf3]
	vqneg.s32	d16, d16
@ CHECK: vqneg.s8	q8, q8          @ encoding: [0xe0,0x07,0xf0,0xf3]
	vqneg.s8	q8, q8
@ CHECK: vqneg.s16	q8, q8          @ encoding: [0xe0,0x07,0xf4,0xf3]
	vqneg.s16	q8, q8
@ CHECK: vqneg.s32	q8, q8          @ encoding: [0xe0,0x07,0xf8,0xf3]
	vqneg.s32	q8, q8
