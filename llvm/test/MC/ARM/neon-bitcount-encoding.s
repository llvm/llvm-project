@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vcnt.8	d16, d16                @ encoding: [0x20,0x05,0xf0,0xf3]
	vcnt.8	d16, d16
@ CHECK: vcnt.8	q8, q8                  @ encoding: [0x60,0x05,0xf0,0xf3]
	vcnt.8	q8, q8
@ CHECK: vclz.i8	d16, d16                @ encoding: [0xa0,0x04,0xf0,0xf3]
	vclz.i8	d16, d16
@ CHECK: vclz.i16	d16, d16        @ encoding: [0xa0,0x04,0xf4,0xf3]
	vclz.i16	d16, d16
@ CHECK: vclz.i32	d16, d16        @ encoding: [0xa0,0x04,0xf8,0xf3]
	vclz.i32	d16, d16
@ CHECK: vclz.i8	q8, q8                  @ encoding: [0xe0,0x04,0xf0,0xf3]
	vclz.i8	q8, q8
@ CHECK: vclz.i16	q8, q8          @ encoding: [0xe0,0x04,0xf4,0xf3]
	vclz.i16	q8, q8
@ CHECK: vclz.i32	q8, q8          @ encoding: [0xe0,0x04,0xf8,0xf3]
	vclz.i32	q8, q8
@ CHECK: vcls.s8	d16, d16                @ encoding: [0x20,0x04,0xf0,0xf3]
	vcls.s8	d16, d16
@ CHECK: vcls.s16	d16, d16        @ encoding: [0x20,0x04,0xf4,0xf3]
	vcls.s16	d16, d16
@ CHECK: vcls.s32	d16, d16        @ encoding: [0x20,0x04,0xf8,0xf3]
	vcls.s32	d16, d16
@ CHECK: vcls.s8	q8, q8                  @ encoding: [0x60,0x04,0xf0,0xf3]
	vcls.s8	q8, q8
@ CHECK: vcls.s16	q8, q8          @ encoding: [0x60,0x04,0xf4,0xf3]
	vcls.s16	q8, q8
@ CHECK: vcls.s32	q8, q8          @ encoding: [0x60,0x04,0xf8,0xf3]
	vcls.s32	q8, q8

