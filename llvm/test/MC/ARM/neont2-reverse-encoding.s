@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vrev64.8	d16, d16        @ encoding: [0xf0,0xff,0x20,0x00]
	vrev64.8	d16, d16
@ CHECK: vrev64.16	d16, d16        @ encoding: [0xf4,0xff,0x20,0x00]
	vrev64.16	d16, d16
@ CHECK: vrev64.32	d16, d16        @ encoding: [0xf8,0xff,0x20,0x00]
	vrev64.32	d16, d16
@ CHECK: vrev64.8	q8, q8          @ encoding: [0xf0,0xff,0x60,0x00]
	vrev64.8	q8, q8
@ CHECK: vrev64.16	q8, q8          @ encoding: [0xf4,0xff,0x60,0x00]
	vrev64.16	q8, q8
@ CHECK: vrev64.32	q8, q8          @ encoding: [0xf8,0xff,0x60,0x00]
	vrev64.32	q8, q8
@ CHECK: vrev32.8	d16, d16        @ encoding: [0xf0,0xff,0xa0,0x00]
	vrev32.8	d16, d16
@ CHECK: vrev32.16	d16, d16        @ encoding: [0xf4,0xff,0xa0,0x00]
	vrev32.16	d16, d16
@ CHECK: vrev32.8	q8, q8          @ encoding: [0xf0,0xff,0xe0,0x00]
	vrev32.8	q8, q8
@ CHECK: vrev32.16	q8, q8          @ encoding: [0xf4,0xff,0xe0,0x00]
	vrev32.16	q8, q8
@ CHECK: vrev16.8	d16, d16        @ encoding: [0xf0,0xff,0x20,0x01]
	vrev16.8	d16, d16
@ CHECK: vrev16.8	q8, q8          @ encoding: [0xf0,0xff,0x60,0x01]
	vrev16.8	q8, q8
