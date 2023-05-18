@RUN: llvm-mc -triple thumbv7-unknown-unknown -mcpu=cortex-a8 -show-encoding < %s | FileCheck %s

.code 16

	vdup.8	d16, r1
	vdup.16	d15, r2
	vdup.32	d14, r3
	vdup.8	q9, r4
	vdup.16	q8, r5
	vdup.32	q7, r6

@ CHECK: vdup.8	d16, r1                 @ encoding: [0xc0,0xee,0x90,0x1b]
@ CHECK: vdup.16	d15, r2         @ encoding: [0x8f,0xee,0x30,0x2b]
@ CHECK: vdup.32	d14, r3         @ encoding: [0x8e,0xee,0x10,0x3b]
@ CHECK: vdup.8	q9, r4                  @ encoding: [0xe2,0xee,0x90,0x4b]
@ CHECK: vdup.16	q8, r5          @ encoding: [0xa0,0xee,0xb0,0x5b]
@ CHECK: vdup.32	q7, r6          @ encoding: [0xae,0xee,0x10,0x6b]

	vdup.8	d16, d11[0]
	vdup.16	d17, d12[0]
	vdup.32	d18, d13[0]
	vdup.8	q3, d10[0]
	vdup.16	q9, d9[0]
	vdup.32	q8, d8[0]
	vdup.8	d16, d11[1]
	vdup.16	d17, d12[1]
	vdup.32	d18, d13[1]
	vdup.8	q3, d10[1]
	vdup.16	q9, d9[1]
	vdup.32	q8, d8[1]

@ CHECK: vdup.8	d16, d11[0]             @ encoding: [0xf1,0xff,0x0b,0x0c]
@ CHECK: vdup.16 d17, d12[0]            @ encoding: [0xf2,0xff,0x0c,0x1c]
@ CHECK: vdup.32 d18, d13[0]            @ encoding: [0xf4,0xff,0x0d,0x2c]
@ CHECK: vdup.8	q3, d10[0]              @ encoding: [0xb1,0xff,0x4a,0x6c]
@ CHECK: vdup.16 q9, d9[0]              @ encoding: [0xf2,0xff,0x49,0x2c]
@ CHECK: vdup.32 q8, d8[0]              @ encoding: [0xf4,0xff,0x48,0x0c]
@ CHECK: vdup.8	d16, d11[1]             @ encoding: [0xf3,0xff,0x0b,0x0c]
@ CHECK: vdup.16 d17, d12[1]            @ encoding: [0xf6,0xff,0x0c,0x1c]
@ CHECK: vdup.32 d18, d13[1]            @ encoding: [0xfc,0xff,0x0d,0x2c]
@ CHECK: vdup.8	q3, d10[1]              @ encoding: [0xb3,0xff,0x4a,0x6c]
@ CHECK: vdup.16 q9, d9[1]              @ encoding: [0xf6,0xff,0x49,0x2c]
@ CHECK: vdup.32 q8, d8[1]              @ encoding: [0xfc,0xff,0x48,0x0c]
