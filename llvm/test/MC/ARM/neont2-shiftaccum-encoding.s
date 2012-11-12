@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

	vsra.s8 d17, d16, #8
	vsra.s16 d15, d14, #16
	vsra.s32 d13, d12, #32
	vsra.s64 d11, d10, #64
	vsra.s8 q7, q2, #8
	vsra.s16 q3, q6, #16
	vsra.s32 q9, q5, #32
	vsra.s64 q8, q4, #64
	vsra.u8 d17, d16, #8
	vsra.u16 d11, d14, #11
	vsra.u32 d12, d15, #22
	vsra.u64 d13, d16, #54
	vsra.u8 q1, q7, #8
	vsra.u16 q2, q7, #6
	vsra.u32 q3, q6, #21
	vsra.u64 q4, q5, #25

        @ Two-operand syntax variant.
	vsra.s8 d16, #8
	vsra.s16 d14, #16
	vsra.s32 d12, #32
	vsra.s64 d10, #64
	vsra.s8 q2, #8
	vsra.s16 q6, #16
	vsra.s32 q5, #32
	vsra.s64 q4, #64
	vsra.u8 d16, #8
	vsra.u16 d14, #11
	vsra.u32 d15, #22
	vsra.u64 d16, #54
	vsra.u8 q7, #8
	vsra.u16 q7, #6
	vsra.u32 q6, #21
	vsra.u64 q5, #25

@ CHECK: vsra.s8	d17, d16, #8    @ encoding: [0xc8,0xef,0x30,0x11]
@ CHECK: vsra.s16	d15, d14, #16   @ encoding: [0x90,0xef,0x1e,0xf1]
@ CHECK: vsra.s32	d13, d12, #32   @ encoding: [0xa0,0xef,0x1c,0xd1]
@ CHECK: vsra.s64	d11, d10, #64   @ encoding: [0x80,0xef,0x9a,0xb1]
@ CHECK: vsra.s8	q7, q2, #8      @ encoding: [0x88,0xef,0x54,0xe1]
@ CHECK: vsra.s16	q3, q6, #16     @ encoding: [0x90,0xef,0x5c,0x61]
@ CHECK: vsra.s32	q9, q5, #32     @ encoding: [0xe0,0xef,0x5a,0x21]
@ CHECK: vsra.s64	q8, q4, #64     @ encoding: [0xc0,0xef,0xd8,0x01]
@ CHECK: vsra.u8	d17, d16, #8    @ encoding: [0xc8,0xff,0x30,0x11]
@ CHECK: vsra.u16	d11, d14, #11   @ encoding: [0x95,0xff,0x1e,0xb1]
@ CHECK: vsra.u32	d12, d15, #22   @ encoding: [0xaa,0xff,0x1f,0xc1]
@ CHECK: vsra.u64	d13, d16, #54   @ encoding: [0x8a,0xff,0xb0,0xd1]
@ CHECK: vsra.u8	q1, q7, #8      @ encoding: [0x88,0xff,0x5e,0x21]
@ CHECK: vsra.u16	q2, q7, #6      @ encoding: [0x9a,0xff,0x5e,0x41]
@ CHECK: vsra.u32	q3, q6, #21     @ encoding: [0xab,0xff,0x5c,0x61]
@ CHECK: vsra.u64	q4, q5, #25     @ encoding: [0xa7,0xff,0xda,0x81]

@ CHECK: vsra.s8	d16, d16, #8            @ encoding: [0xc8,0xef,0x30,0x01]
@ CHECK: vsra.s16	d14, d14, #16   @ encoding: [0x90,0xef,0x1e,0xe1]
@ CHECK: vsra.s32	d12, d12, #32   @ encoding: [0xa0,0xef,0x1c,0xc1]
@ CHECK: vsra.s64	d10, d10, #64   @ encoding: [0x80,0xef,0x9a,0xa1]
@ CHECK: vsra.s8	q2, q2, #8              @ encoding: [0x88,0xef,0x54,0x41]
@ CHECK: vsra.s16	q6, q6, #16     @ encoding: [0x90,0xef,0x5c,0xc1]
@ CHECK: vsra.s32	q5, q5, #32     @ encoding: [0xa0,0xef,0x5a,0xa1]
@ CHECK: vsra.s64	q4, q4, #64     @ encoding: [0x80,0xef,0xd8,0x81]
@ CHECK: vsra.u8	d16, d16, #8            @ encoding: [0xc8,0xff,0x30,0x01]
@ CHECK: vsra.u16	d14, d14, #11   @ encoding: [0x95,0xff,0x1e,0xe1]
@ CHECK: vsra.u32	d15, d15, #22   @ encoding: [0xaa,0xff,0x1f,0xf1]
@ CHECK: vsra.u64	d16, d16, #54   @ encoding: [0xca,0xff,0xb0,0x01]
@ CHECK: vsra.u8	q7, q7, #8              @ encoding: [0x88,0xff,0x5e,0xe1]
@ CHECK: vsra.u16	q7, q7, #6      @ encoding: [0x9a,0xff,0x5e,0xe1]
@ CHECK: vsra.u32	q6, q6, #21     @ encoding: [0xab,0xff,0x5c,0xc1]
@ CHECK: vsra.u64	q5, q5, #25     @ encoding: [0xa7,0xff,0xda,0xa1]


	vrsra.s8 d5, d26, #8
	vrsra.s16 d6, d25, #16
	vrsra.s32 d7, d24, #32
	vrsra.s64 d14, d23, #64
	vrsra.u8 d15, d22, #8
	vrsra.u16 d16, d21, #16
	vrsra.u32 d17, d20, #32
	vrsra.u64 d18, d19, #64
	vrsra.s8 q1, q2, #8
	vrsra.s16 q2, q3, #16
	vrsra.s32 q3, q4, #32
	vrsra.s64 q4, q5, #64
	vrsra.u8 q5, q6, #8
	vrsra.u16 q6, q7, #16
	vrsra.u32 q7, q8, #32
	vrsra.u64 q8, q9, #64

        @ Two-operand syntax variant.
	vrsra.s8 d26, #8
	vrsra.s16 d25, #16
	vrsra.s32 d24, #32
	vrsra.s64 d23, #64
	vrsra.u8 d22, #8
	vrsra.u16 d21, #16
	vrsra.u32 d20, #32
	vrsra.u64 d19, #64
	vrsra.s8 q2, #8
	vrsra.s16 q3, #16
	vrsra.s32 q4, #32
	vrsra.s64 q5, #64
	vrsra.u8 q6, #8
	vrsra.u16 q7, #16
	vrsra.u32 q8, #32
	vrsra.u64 q9, #64

@ CHECK: vrsra.s8	d5, d26, #8     @ encoding: [0x88,0xef,0x3a,0x53]
@ CHECK: vrsra.s16	d6, d25, #16    @ encoding: [0x90,0xef,0x39,0x63]
@ CHECK: vrsra.s32	d7, d24, #32    @ encoding: [0xa0,0xef,0x38,0x73]
@ CHECK: vrsra.s64	d14, d23, #64   @ encoding: [0x80,0xef,0xb7,0xe3]
@ CHECK: vrsra.u8	d15, d22, #8    @ encoding: [0x88,0xff,0x36,0xf3]
@ CHECK: vrsra.u16	d16, d21, #16   @ encoding: [0xd0,0xff,0x35,0x03]
@ CHECK: vrsra.u32	d17, d20, #32   @ encoding: [0xe0,0xff,0x34,0x13]
@ CHECK: vrsra.u64	d18, d19, #64   @ encoding: [0xc0,0xff,0xb3,0x23]
@ CHECK: vrsra.s8	q1, q2, #8      @ encoding: [0x88,0xef,0x54,0x23]
@ CHECK: vrsra.s16	q2, q3, #16     @ encoding: [0x90,0xef,0x56,0x43]
@ CHECK: vrsra.s32	q3, q4, #32     @ encoding: [0xa0,0xef,0x58,0x63]
@ CHECK: vrsra.s64	q4, q5, #64     @ encoding: [0x80,0xef,0xda,0x83]
@ CHECK: vrsra.u8	q5, q6, #8      @ encoding: [0x88,0xff,0x5c,0xa3]
@ CHECK: vrsra.u16	q6, q7, #16     @ encoding: [0x90,0xff,0x5e,0xc3]
@ CHECK: vrsra.u32	q7, q8, #32     @ encoding: [0xa0,0xff,0x70,0xe3]
@ CHECK: vrsra.u64	q8, q9, #64     @ encoding: [0xc0,0xff,0xf2,0x03]

@ CHECK: vrsra.s8	d26, d26, #8    @ encoding: [0xc8,0xef,0x3a,0xa3]
@ CHECK: vrsra.s16	d25, d25, #16   @ encoding: [0xd0,0xef,0x39,0x93]
@ CHECK: vrsra.s32	d24, d24, #32   @ encoding: [0xe0,0xef,0x38,0x83]
@ CHECK: vrsra.s64	d23, d23, #64   @ encoding: [0xc0,0xef,0xb7,0x73]
@ CHECK: vrsra.u8	d22, d22, #8    @ encoding: [0xc8,0xff,0x36,0x63]
@ CHECK: vrsra.u16	d21, d21, #16   @ encoding: [0xd0,0xff,0x35,0x53]
@ CHECK: vrsra.u32	d20, d20, #32   @ encoding: [0xe0,0xff,0x34,0x43]
@ CHECK: vrsra.u64	d19, d19, #64   @ encoding: [0xc0,0xff,0xb3,0x33]
@ CHECK: vrsra.s8	q2, q2, #8      @ encoding: [0x88,0xef,0x54,0x43]
@ CHECK: vrsra.s16	q3, q3, #16     @ encoding: [0x90,0xef,0x56,0x63]
@ CHECK: vrsra.s32	q4, q4, #32     @ encoding: [0xa0,0xef,0x58,0x83]
@ CHECK: vrsra.s64	q5, q5, #64     @ encoding: [0x80,0xef,0xda,0xa3]
@ CHECK: vrsra.u8	q6, q6, #8      @ encoding: [0x88,0xff,0x5c,0xc3]
@ CHECK: vrsra.u16	q7, q7, #16     @ encoding: [0x90,0xff,0x5e,0xe3]
@ CHECK: vrsra.u32	q8, q8, #32     @ encoding: [0xe0,0xff,0x70,0x03]
@ CHECK: vrsra.u64	q9, q9, #64     @ encoding: [0xc0,0xff,0xf2,0x23]


	vsli.8 d11, d12, #7
	vsli.16 d12, d13, #15
	vsli.32 d13, d14, #31
	vsli.64 d14, d15, #63
	vsli.8 q1, q8, #7
	vsli.16 q2, q7, #15
	vsli.32 q3, q4, #31
	vsli.64 q4, q5, #63
	vsri.8 d28, d11, #8
	vsri.16 d26, d12, #16
	vsri.32 d24, d13, #32
	vsri.64 d21, d14, #64
	vsri.8 q1, q8, #8
	vsri.16 q5, q2, #16
	vsri.32 q7, q4, #32
	vsri.64 q9, q6, #64

        @ Two-operand syntax variant.
	vsli.8 d12, #7
	vsli.16 d13, #15
	vsli.32 d14, #31
	vsli.64 d15, #63
	vsli.8 q8, #7
	vsli.16 q7, #15
	vsli.32 q4, #31
	vsli.64 q5, #63
	vsri.8 d11, #8
	vsri.16 d12, #16
	vsri.32 d13, #32
	vsri.64 d14, #64
	vsri.8 q8, #8
	vsri.16 q2, #16
	vsri.32 q4, #32
	vsri.64 q6, #64

@ CHECK: vsli.8	d11, d12, #7            @ encoding: [0x8f,0xff,0x1c,0xb5]
@ CHECK: vsli.16	d12, d13, #15   @ encoding: [0x9f,0xff,0x1d,0xc5]
@ CHECK: vsli.32	d13, d14, #31   @ encoding: [0xbf,0xff,0x1e,0xd5]
@ CHECK: vsli.64	d14, d15, #63   @ encoding: [0xbf,0xff,0x9f,0xe5]
@ CHECK: vsli.8	q1, q8, #7              @ encoding: [0x8f,0xff,0x70,0x25]
@ CHECK: vsli.16	q2, q7, #15     @ encoding: [0x9f,0xff,0x5e,0x45]
@ CHECK: vsli.32	q3, q4, #31     @ encoding: [0xbf,0xff,0x58,0x65]
@ CHECK: vsli.64	q4, q5, #63     @ encoding: [0xbf,0xff,0xda,0x85]
@ CHECK: vsri.8	d28, d11, #8            @ encoding: [0xc8,0xff,0x1b,0xc4]
@ CHECK: vsri.16	d26, d12, #16   @ encoding: [0xd0,0xff,0x1c,0xa4]
@ CHECK: vsri.32	d24, d13, #32   @ encoding: [0xe0,0xff,0x1d,0x84]
@ CHECK: vsri.64	d21, d14, #64   @ encoding: [0xc0,0xff,0x9e,0x54]
@ CHECK: vsri.8	q1, q8, #8              @ encoding: [0x88,0xff,0x70,0x24]
@ CHECK: vsri.16	q5, q2, #16     @ encoding: [0x90,0xff,0x54,0xa4]
@ CHECK: vsri.32	q7, q4, #32     @ encoding: [0xa0,0xff,0x58,0xe4]
@ CHECK: vsri.64	q9, q6, #64     @ encoding: [0xc0,0xff,0xdc,0x24]

@ CHECK: vsli.8	d12, d12, #7            @ encoding: [0x8f,0xff,0x1c,0xc5]
@ CHECK: vsli.16	d13, d13, #15           @ encoding: [0x9f,0xff,0x1d,0xd5]
@ CHECK: vsli.32	d14, d14, #31           @ encoding: [0xbf,0xff,0x1e,0xe5]
@ CHECK: vsli.64	d15, d15, #63           @ encoding: [0xbf,0xff,0x9f,0xf5]
@ CHECK: vsli.8	q8, q8, #7              @ encoding: [0xcf,0xff,0x70,0x05]
@ CHECK: vsli.16	q7, q7, #15             @ encoding: [0x9f,0xff,0x5e,0xe5]
@ CHECK: vsli.32	q4, q4, #31             @ encoding: [0xbf,0xff,0x58,0x85]
@ CHECK: vsli.64	q5, q5, #63             @ encoding: [0xbf,0xff,0xda,0xa5]
@ CHECK: vsri.8	d11, d11, #8            @ encoding: [0x88,0xff,0x1b,0xb4]
@ CHECK: vsri.16	d12, d12, #16           @ encoding: [0x90,0xff,0x1c,0xc4]
@ CHECK: vsri.32	d13, d13, #32           @ encoding: [0xa0,0xff,0x1d,0xd4]
@ CHECK: vsri.64	d14, d14, #64           @ encoding: [0x80,0xff,0x9e,0xe4]
@ CHECK: vsri.8	q8, q8, #8              @ encoding: [0xc8,0xff,0x70,0x04]
@ CHECK: vsri.16	q2, q2, #16             @ encoding: [0x90,0xff,0x54,0x44]
@ CHECK: vsri.32	q4, q4, #32             @ encoding: [0xa0,0xff,0x58,0x84]
@ CHECK: vsri.64	q6, q6, #64             @ encoding: [0x80,0xff,0xdc,0xc4]
