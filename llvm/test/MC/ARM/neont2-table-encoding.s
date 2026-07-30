@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vtbl.8	d16, {d17}, d16
	vtbl.8	d16, {d16, d17}, d18
	vtbl.8	d16, {d16, d17, d18}, d20
	vtbl.8	d16, {d16, d17, d18, d19}, d20

@ CHECK: vtbl.8	d16, {d17}, d16         @ encoding: [0xf1,0xff,0xa0,0x08]
@ CHECK: vtbl.8	d16, {d16, d17}, d18    @ encoding: [0xf0,0xff,0xa2,0x09]
@ CHECK: vtbl.8	d16, {d16, d17, d18}, d20 @ encoding: [0xf0,0xff,0xa4,0x0a]
@ CHECK: vtbl.8	d16, {d16, d17, d18, d19}, d20 @ encoding: [0xf0,0xff,0xa4,0x0b]


	vtbx.8	d18, {d16}, d17
	vtbx.8	d19, {d16, d17}, d18
	vtbx.8	d20, {d16, d17, d18}, d21
	vtbx.8	d20, {d16, d17, d18, d19}, d21

@ CHECK: vtbx.8	d18, {d16}, d17         @ encoding: [0xf0,0xff,0xe1,0x28]
@ CHECK: vtbx.8	d19, {d16, d17}, d18    @ encoding: [0xf0,0xff,0xe2,0x39]
@ CHECK: vtbx.8	d20, {d16, d17, d18}, d21 @ encoding: [0xf0,0xff,0xe5,0x4a]
@ CHECK: vtbx.8	d20, {d16, d17, d18, d19}, d21 @ encoding: [0xf0,0xff,0xe5,0x4b]
