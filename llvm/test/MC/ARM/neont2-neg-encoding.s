@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vneg.s8	d16, d16                @ encoding: [0xf1,0xff,0xa0,0x03]
	vneg.s8	d16, d16
@ CHECK: vneg.s16	d16, d16        @ encoding: [0xf5,0xff,0xa0,0x03]
	vneg.s16	d16, d16
@ CHECK: vneg.s32	d16, d16        @ encoding: [0xf9,0xff,0xa0,0x03]
	vneg.s32	d16, d16
@ CHECK: vneg.f32	d16, d16        @ encoding: [0xf9,0xff,0xa0,0x07]
	vneg.f32	d16, d16
@ CHECK: vneg.s8	q8, q8                  @ encoding: [0xf1,0xff,0xe0,0x03]
	vneg.s8	q8, q8
@ CHECK: vneg.s16	q8, q8          @ encoding: [0xf5,0xff,0xe0,0x03]
	vneg.s16	q8, q8
@ CHECK: vneg.s32	q8, q8          @ encoding: [0xf9,0xff,0xe0,0x03]
	vneg.s32	q8, q8
@ CHECK: vneg.f32	q8, q8          @ encoding: [0xf9,0xff,0xe0,0x07]
	vneg.f32	q8, q8
@ CHECK: vqneg.s8	d16, d16        @ encoding: [0xf0,0xff,0xa0,0x07]
	vqneg.s8	d16, d16
@ CHECK: vqneg.s16	d16, d16        @ encoding: [0xf4,0xff,0xa0,0x07]
	vqneg.s16	d16, d16
@ CHECK: vqneg.s32	d16, d16        @ encoding: [0xf8,0xff,0xa0,0x07]
	vqneg.s32	d16, d16
@ CHECK: vqneg.s8	q8, q8          @ encoding: [0xf0,0xff,0xe0,0x07]
	vqneg.s8	q8, q8
@ CHECK: vqneg.s16	q8, q8          @ encoding: [0xf4,0xff,0xe0,0x07]
	vqneg.s16	q8, q8
@ CHECK: vqneg.s32	q8, q8          @ encoding: [0xf8,0xff,0xe0,0x07]
	vqneg.s32	q8, q8
