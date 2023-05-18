@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vand	d16, d17, d16
	vand	q8, q8, q9

@ CHECK: vand	d16, d17, d16           @ encoding: [0x41,0xef,0xb0,0x01]
@ CHECK: vand	q8, q8, q9              @ encoding: [0x40,0xef,0xf2,0x01]

	veor	d16, d17, d16
	veor	q8, q8, q9

@ CHECK: veor	d16, d17, d16           @ encoding: [0x41,0xff,0xb0,0x01]
@ CHECK: veor	q8, q8, q9              @ encoding: [0x40,0xff,0xf2,0x01]


	vorr	d16, d17, d16
	vorr	q8, q8, q9
@	vorr.i32	d16, #0x1000000
@	vorr.i32	q8, #0x1000000
@	vorr.i32	q8, #0x0

@ CHECK: vorr	d16, d17, d16           @ encoding: [0x61,0xef,0xb0,0x01]
@ CHECK: vorr	q8, q8, q9              @ encoding: [0x60,0xef,0xf2,0x01]


	vbic	d16, d17, d16
	vbic	q8, q8, q9
@	vbic.i32	d16, #0xFF000000
@	vbic.i32	q8, #0xFF000000

@ CHECK: vbic	d16, d17, d16           @ encoding: [0x51,0xef,0xb0,0x01]
@ CHECK: vbic	q8, q8, q9              @ encoding: [0x50,0xef,0xf2,0x01]


	vorn	d16, d17, d16
	vorn	q8, q8, q9

@ CHECK: vorn	d16, d17, d16           @ encoding: [0x71,0xef,0xb0,0x01]
@ CHECK: vorn	q8, q8, q9              @ encoding: [0x70,0xef,0xf2,0x01]


	vmvn	d16, d16
	vmvn	q8, q8

@ CHECK: vmvn	d16, d16                @ encoding: [0xf0,0xff,0xa0,0x05]
@ CHECK: vmvn	q8, q8                  @ encoding: [0xf0,0xff,0xe0,0x05]


	vbsl	d18, d17, d16
	vbsl	q8, q10, q9
	vbit	d18, d17, d16
	vbit	q8, q10, q9
	vbif	d18, d17, d16
	vbif	q8, q10, q9

@ CHECK: vbsl	d18, d17, d16           @ encoding: [0x51,0xff,0xb0,0x21]
@ CHECK: vbsl	q8, q10, q9             @ encoding: [0x54,0xff,0xf2,0x01]
@ CHECK: vbit	d18, d17, d16           @ encoding: [0x61,0xff,0xb0,0x21]
@ CHECK: vbit	q8, q10, q9             @ encoding: [0x64,0xff,0xf2,0x01]
@ CHECK: vbif	d18, d17, d16           @ encoding: [0x71,0xff,0xb0,0x21]
@ CHECK: vbif	q8, q10, q9             @ encoding: [0x74,0xff,0xf2,0x01]
