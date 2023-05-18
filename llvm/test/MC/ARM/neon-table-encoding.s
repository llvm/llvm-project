@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

	vtbl.8	d16, {d17}, d16
	vtbl.8	d16, {d16, d17}, d18
	vtbl.8	d16, {d16, d17, d18}, d20
	vtbl.8	d16, {d16, d17, d18, d19}, d20

@ CHECK: vtbl.8	d16, {d17}, d16         @ encoding: [0xa0,0x08,0xf1,0xf3]
@ CHECK: vtbl.8	d16, {d16, d17}, d18    @ encoding: [0xa2,0x09,0xf0,0xf3]
@ CHECK: vtbl.8	d16, {d16, d17, d18}, d20 @ encoding: [0xa4,0x0a,0xf0,0xf3]
@ CHECK: vtbl.8	d16, {d16, d17, d18, d19}, d20 @ encoding: [0xa4,0x0b,0xf0,0xf3]


	vtbx.8	d18, {d16}, d17
	vtbx.8	d19, {d16, d17}, d18
	vtbx.8	d20, {d16, d17, d18}, d21
	vtbx.8	d20, {d16, d17, d18, d19}, d21

@ CHECK: vtbx.8	d18, {d16}, d17         @ encoding: [0xe1,0x28,0xf0,0xf3]
@ CHECK: vtbx.8	d19, {d16, d17}, d18    @ encoding: [0xe2,0x39,0xf0,0xf3]
@ CHECK: vtbx.8	d20, {d16, d17, d18}, d21 @ encoding: [0xe5,0x4a,0xf0,0xf3]
@ CHECK: vtbx.8	d20, {d16, d17, d18, d19}, d21 @ encoding: [0xe5,0x4b,0xf0,0xf3]
