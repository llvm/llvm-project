@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vabs.s8	d16, d16                @ encoding: [0xf1,0xff,0x20,0x03]
	vabs.s8	d16, d16
@ CHECK: vabs.s16	d16, d16        @ encoding: [0xf5,0xff,0x20,0x03]
	vabs.s16	d16, d16
@ CHECK: vabs.s32	d16, d16        @ encoding: [0xf9,0xff,0x20,0x03]
	vabs.s32	d16, d16
@ CHECK: vabs.f32	d16, d16        @ encoding: [0xf9,0xff,0x20,0x07]
	vabs.f32	d16, d16
@ CHECK: vabs.s8	q8, q8                  @ encoding: [0xf1,0xff,0x60,0x03]
	vabs.s8	q8, q8
@ CHECK: vabs.s16	q8, q8          @ encoding: [0xf5,0xff,0x60,0x03]
	vabs.s16	q8, q8
@ CHECK: vabs.s32	q8, q8          @ encoding: [0xf9,0xff,0x60,0x03]
	vabs.s32	q8, q8
@ CHECK: vabs.f32	q8, q8          @ encoding: [0xf9,0xff,0x60,0x07]
	vabs.f32	q8, q8

@ CHECK: vqabs.s8	d16, d16        @ encoding: [0xf0,0xff,0x20,0x07]
	vqabs.s8	d16, d16
@ CHECK: vqabs.s16	d16, d16        @ encoding: [0xf4,0xff,0x20,0x07]
	vqabs.s16	d16, d16
@ CHECK: vqabs.s32	d16, d16        @ encoding: [0xf8,0xff,0x20,0x07]
	vqabs.s32	d16, d16
@ CHECK: vqabs.s8	q8, q8          @ encoding: [0xf0,0xff,0x60,0x07]
	vqabs.s8	q8, q8
@ CHECK: vqabs.s16	q8, q8          @ encoding: [0xf4,0xff,0x60,0x07]
	vqabs.s16	q8, q8
@ CHECK: vqabs.s32	q8, q8          @ encoding: [0xf8,0xff,0x60,0x07]
	vqabs.s32	q8, q8
