@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vdup.8	d16, r0
	vdup.16	d16, r0
	vdup.32	d16, r0

@ CHECK: vdup.8	d16, r0                 @ encoding: [0x90,0x0b,0xc0,0xee]
@ CHECK: vdup.16	d16, r0         @ encoding: [0xb0,0x0b,0x80,0xee]
@ CHECK: vdup.32	d16, r0         @ encoding: [0x90,0x0b,0x80,0xee]

	vdup.8	q8, r0
	vdup.16	q8, r0
	vdup.32	q8, r0

@ CHECK: vdup.8	q8, r0                  @ encoding: [0x90,0x0b,0xe0,0xee]
@ CHECK: vdup.16	q8, r0          @ encoding: [0xb0,0x0b,0xa0,0xee]
@ CHECK: vdup.32	q8, r0          @ encoding: [0x90,0x0b,0xa0,0xee]

	vdup.8	d16, d16[1]
	vdup.16	d16, d16[1]
	vdup.32	d16, d16[1]

@ CHECK: vdup.8	d16, d16[1]             @ encoding: [0x20,0x0c,0xf3,0xf3]
@ CHECK: vdup.16	d16, d16[1]     @ encoding: [0x20,0x0c,0xf6,0xf3]
@ CHECK: vdup.32	d16, d16[1]     @ encoding: [0x20,0x0c,0xfc,0xf3]

	vdup.8	q8, d16[1]
	vdup.16	q8, d16[1]
	vdup.32	q8, d16[1]

@ CHECK: vdup.8	q8, d16[1]              @ encoding: [0x60,0x0c,0xf3,0xf3]
@ CHECK: vdup.16	q8, d16[1]      @ encoding: [0x60,0x0c,0xf6,0xf3]
@ CHECK: vdup.32	q8, d16[1]      @ encoding: [0x60,0x0c,0xfc,0xf3]
