@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

@ CHECK: vext.8	d16, d17, d16, #3       @ encoding: [0xf1,0xef,0xa0,0x03]
	vext.8	d16, d17, d16, #3
@ CHECK: vext.8	d16, d17, d16, #5       @ encoding: [0xf1,0xef,0xa0,0x05]
	vext.8	d16, d17, d16, #5
@ CHECK: vext.8	q8, q9, q8, #3          @ encoding: [0xf2,0xef,0xe0,0x03]
	vext.8	q8, q9, q8, #3
@ CHECK: vext.8	q8, q9, q8, #7          @ encoding: [0xf2,0xef,0xe0,0x07]
	vext.8	q8, q9, q8, #7
@ CHECK: vext.16	d16, d17, d16, #3       @ encoding: [0xf1,0xef,0xa0,0x06]
	vext.16	d16, d17, d16, #3
@ CHECK: vext.32	q8, q9, q8, #3          @ encoding: [0xf2,0xef,0xe0,0x0c]
	vext.32	q8, q9, q8, #3
@ CHECK: vtrn.8	d17, d16                @ encoding: [0xf2,0xff,0xa0,0x10]
	vtrn.8	d17, d16
@ CHECK: vtrn.16	d17, d16                @ encoding: [0xf6,0xff,0xa0,0x10]
	vtrn.16	d17, d16
@ CHECK: vtrn.32	d17, d16                @ encoding: [0xfa,0xff,0xa0,0x10]
	vtrn.32	d17, d16
@ CHECK: vtrn.8	q9, q8                  @ encoding: [0xf2,0xff,0xe0,0x20]
	vtrn.8	q9, q8
@ CHECK: vtrn.16	q9, q8                  @ encoding: [0xf6,0xff,0xe0,0x20]
	vtrn.16	q9, q8
@ CHECK: vtrn.32	q9, q8                  @ encoding: [0xfa,0xff,0xe0,0x20]
	vtrn.32	q9, q8
@ CHECK: vuzp.8	d17, d16                @ encoding: [0xf2,0xff,0x20,0x11]
	vuzp.8	d17, d16
@ CHECK: vuzp.16	d17, d16                @ encoding: [0xf6,0xff,0x20,0x11]
	vuzp.16	d17, d16
@ CHECK: vuzp.8	q9, q8                  @ encoding: [0xf2,0xff,0x60,0x21]
	vuzp.8	q9, q8
@ CHECK: vuzp.16	q9, q8                  @ encoding: [0xf6,0xff,0x60,0x21]
	vuzp.16	q9, q8
@ CHECK: vuzp.32	q9, q8                  @ encoding: [0xfa,0xff,0x60,0x21]
	vuzp.32	q9, q8
@ CHECK: vzip.8	d17, d16                @ encoding: [0xf2,0xff,0xa0,0x11]
	vzip.8	d17, d16
@ CHECK: vzip.16	d17, d16                @ encoding: [0xf6,0xff,0xa0,0x11]
	vzip.16	d17, d16
@ CHECK: vzip.8	q9, q8                  @ encoding: [0xf2,0xff,0xe0,0x21]
	vzip.8	q9, q8
@ CHECK: vzip.16	q9, q8                  @ encoding: [0xf6,0xff,0xe0,0x21]
	vzip.16	q9, q8
@ CHECK: vzip.32	q9, q8                  @ encoding: [0xfa,0xff,0xe0,0x21]
	vzip.32	q9, q8
