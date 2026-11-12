@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vcnt.8	d16, d16
	vcnt.8	q8, q8

@ CHECK: vcnt.8	d16, d16                @ encoding: [0xf0,0xff,0x20,0x05]
@ CHECK: vcnt.8	q8, q8                  @ encoding: [0xf0,0xff,0x60,0x05]

	vclz.i8	d16, d16
	vclz.i16	d16, d16
	vclz.i32	d16, d16
	vclz.i8	q8, q8
	vclz.i16	q8, q8
	vclz.i32	q8, q8

@ CHECK: vclz.i8	d16, d16        @ encoding: [0xf0,0xff,0xa0,0x04]
@ CHECK: vclz.i16	d16, d16        @ encoding: [0xf4,0xff,0xa0,0x04]
@ CHECK: vclz.i32	d16, d16        @ encoding: [0xf8,0xff,0xa0,0x04]
@ CHECK: vclz.i8	q8, q8          @ encoding: [0xf0,0xff,0xe0,0x04]
@ CHECK: vclz.i16	q8, q8          @ encoding: [0xf4,0xff,0xe0,0x04]
@ CHECK: vclz.i32	q8, q8          @ encoding: [0xf8,0xff,0xe0,0x04]

	vcls.s8	d16, d16
	vcls.s16	d16, d16
	vcls.s32	d16, d16
	vcls.s8	q8, q8
	vcls.s16	q8, q8
	vcls.s32	q8, q8

@ CHECK: vcls.s8	d16, d16        @ encoding: [0xf0,0xff,0x20,0x04]
@ CHECK: vcls.s16	d16, d16        @ encoding: [0xf4,0xff,0x20,0x04]
@ CHECK: vcls.s32	d16, d16        @ encoding: [0xf8,0xff,0x20,0x04]
@ CHECK: vcls.s8	q8, q8          @ encoding: [0xf0,0xff,0x60,0x04]
@ CHECK: vcls.s16	q8, q8          @ encoding: [0xf4,0xff,0x60,0x04]
@ CHECK: vcls.s32	q8, q8          @ encoding: [0xf8,0xff,0x60,0x04]

