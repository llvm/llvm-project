@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vabs.s8	d16, d16                @ encoding: [0x20,0x03,0xf1,0xf3]
	vabs.s8	d16, d16
@ CHECK: vabs.s16	d16, d16        @ encoding: [0x20,0x03,0xf5,0xf3]
	vabs.s16	d16, d16
@ CHECK: vabs.s32	d16, d16        @ encoding: [0x20,0x03,0xf9,0xf3]
	vabs.s32	d16, d16
@ CHECK: vabs.f32	d16, d16        @ encoding: [0x20,0x07,0xf9,0xf3]
	vabs.f32	d16, d16
@ CHECK: vabs.s8	q8, q8                  @ encoding: [0x60,0x03,0xf1,0xf3]
	vabs.s8	q8, q8
@ CHECK: vabs.s16	q8, q8          @ encoding: [0x60,0x03,0xf5,0xf3]
	vabs.s16	q8, q8
@ CHECK: vabs.s32	q8, q8          @ encoding: [0x60,0x03,0xf9,0xf3]
	vabs.s32	q8, q8
@ CHECK: vabs.f32	q8, q8          @ encoding: [0x60,0x07,0xf9,0xf3]
	vabs.f32	q8, q8

@ CHECK: vqabs.s8	d16, d16        @ encoding: [0x20,0x07,0xf0,0xf3]
	vqabs.s8	d16, d16
@ CHECK: vqabs.s16	d16, d16        @ encoding: [0x20,0x07,0xf4,0xf3]
	vqabs.s16	d16, d16
@ CHECK: vqabs.s32	d16, d16        @ encoding: [0x20,0x07,0xf8,0xf3]
	vqabs.s32	d16, d16
@ CHECK: vqabs.s8	q8, q8          @ encoding: [0x60,0x07,0xf0,0xf3]
	vqabs.s8	q8, q8
@ CHECK: vqabs.s16	q8, q8          @ encoding: [0x60,0x07,0xf4,0xf3]
	vqabs.s16	q8, q8
@ CHECK: vqabs.s32	q8, q8          @ encoding: [0x60,0x07,0xf8,0xf3]
	vqabs.s32	q8, q8
