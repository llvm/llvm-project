@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7-apple-darwin -show-encoding < %s | FileCheck %s

	vld1.8	{d16}, [r0, :64]
	vld1.16	{d16}, [r0]
	vld1.32	{d16}, [r0]
	vld1.64	{d16}, [r0]
	vld1.8	{d16, d17}, [r0, :64]
	vld1.16	{d16, d17}, [r0, :128]
	vld1.32	{d16, d17}, [r0]
	vld1.64	{d16, d17}, [r0]
	vld1.8 {d1, d2, d3}, [r3]
	vld1.16 {d4, d5, d6}, [r3, :64]
	vld1.32 {d5, d6, d7}, [r3]
	vld1.64 {d6, d7, d8}, [r3, :64]
	vld1.8 {d1, d2, d3, d4}, [r3]
	vld1.16 {d4, d5, d6, d7}, [r3, :64]
	vld1.32 {d5, d6, d7, d8}, [r3]
	vld1.64 {d6, d7, d8, d9}, [r3, :64]

	vld1.8	{d16}, [r0, :64]!
	vld1.16	{d16}, [r0]!
	vld1.32	{d16}, [r0]!
	vld1.64	{d16}, [r0]!
	vld1.8	{d16, d17}, [r0, :64]!
	vld1.16	{d16, d17}, [r0, :128]!
	vld1.32	{d16, d17}, [r0]!
	vld1.64	{d16, d17}, [r0]!

	vld1.8	{d16}, [r0, :64], r5
	vld1.16	{d16}, [r0], r5
	vld1.32	{d16}, [r0], r5
	vld1.64	{d16}, [r0], r5
	vld1.8	{d16, d17}, [r0, :64], r5
	vld1.16	{d16, d17}, [r0, :128], r5
	vld1.32	{d16, d17}, [r0], r5
	vld1.64	{d16, d17}, [r0], r5

	vld1.8 {d1, d2, d3}, [r3]!
	vld1.16 {d4, d5, d6}, [r3, :64]!
	vld1.32 {d5, d6, d7}, [r3]!
	vld1.64 {d6, d7, d8}, [r3, :64]!

	vld1.8 {d1, d2, d3}, [r3], r6
	vld1.16 {d4, d5, d6}, [r3, :64], r6
	vld1.32 {d5, d6, d7}, [r3], r6
	vld1.64 {d6, d7, d8}, [r3, :64], r6

	vld1.8 {d1, d2, d3, d4}, [r3]!
	vld1.16 {d4, d5, d6, d7}, [r3, :64]!
	vld1.32 {d5, d6, d7, d8}, [r3]!
	vld1.64 {d6, d7, d8, d9}, [r3, :64]!

	vld1.8 {d1, d2, d3, d4}, [r3], r8
	vld1.16 {d4, d5, d6, d7}, [r3, :64], r8
	vld1.32 {d5, d6, d7, d8}, [r3], r8
	vld1.64 {d6, d7, d8, d9}, [r3, :64], r8

@ CHECK: vld1.8 {d16}, [r0, :64]        @ encoding: [0x1f,0x07,0x60,0xf4]
@ CHECK: vld1.16 {d16}, [r0]            @ encoding: [0x4f,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0]            @ encoding: [0x8f,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0]            @ encoding: [0xcf,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64]   @ encoding: [0x1f,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0]       @ encoding: [0x8f,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0]       @ encoding: [0xcf,0x0a,0x60,0xf4]
@ CHECK: vld1.8 {d1, d2, d3}, [r3]      @ encoding: [0x0f,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64] @ encoding: [0x5f,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3]     @ encoding: [0x8f,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64] @ encoding: [0xdf,0x66,0x23,0xf4]
@ CHECK: vld1.8 {d1, d2, d3, d4}, [r3]  @ encoding: [0x0f,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64] @ encoding: [0x5f,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3]  @ encoding: [0x8f,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64] @ encoding: [0xdf,0x62,0x23,0xf4]
@ CHECK: vld1.8	{d16}, [r0, :64]!       @ encoding: [0x1d,0x07,0x60,0xf4]

@ CHECK: vld1.16 {d16}, [r0]!           @ encoding: [0x4d,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0]!           @ encoding: [0x8d,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0]!           @ encoding: [0xcd,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64]!  @ encoding: [0x1d,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128]! @ encoding: [0x6d,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0]!      @ encoding: [0x8d,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0]!      @ encoding: [0xcd,0x0a,0x60,0xf4]

@ CHECK: vld1.8 {d16}, [r0, :64], r5    @ encoding: [0x15,0x07,0x60,0xf4]
@ CHECK: vld1.16 {d16}, [r0], r5        @ encoding: [0x45,0x07,0x60,0xf4]
@ CHECK: vld1.32 {d16}, [r0], r5        @ encoding: [0x85,0x07,0x60,0xf4]
@ CHECK: vld1.64 {d16}, [r0], r5        @ encoding: [0xc5,0x07,0x60,0xf4]
@ CHECK: vld1.8 {d16, d17}, [r0, :64], r5 @ encoding: [0x15,0x0a,0x60,0xf4]
@ CHECK: vld1.16 {d16, d17}, [r0, :128], r5 @ encoding: [0x65,0x0a,0x60,0xf4]
@ CHECK: vld1.32 {d16, d17}, [r0], r5   @ encoding: [0x85,0x0a,0x60,0xf4]
@ CHECK: vld1.64 {d16, d17}, [r0], r5   @ encoding: [0xc5,0x0a,0x60,0xf4]

@ CHECK: vld1.8	{d1, d2, d3}, [r3]!     @ encoding: [0x0d,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64]! @ encoding: [0x5d,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3]!     @ encoding: [0x8d,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64]! @ encoding: [0xdd,0x66,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3}, [r3], r6  @ encoding: [0x06,0x16,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6}, [r3, :64], r6 @ encoding: [0x56,0x46,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7}, [r3], r6  @ encoding: [0x86,0x56,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8}, [r3, :64], r6 @ encoding: [0xd6,0x66,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3, d4}, [r3]! @ encoding: [0x0d,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64]! @ encoding: [0x5d,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3]! @ encoding: [0x8d,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64]! @ encoding: [0xdd,0x62,0x23,0xf4]

@ CHECK: vld1.8	{d1, d2, d3, d4}, [r3], r8 @ encoding: [0x08,0x12,0x23,0xf4]
@ CHECK: vld1.16 {d4, d5, d6, d7}, [r3, :64], r8 @ encoding: [0x58,0x42,0x23,0xf4]
@ CHECK: vld1.32 {d5, d6, d7, d8}, [r3], r8 @ encoding: [0x88,0x52,0x23,0xf4]
@ CHECK: vld1.64 {d6, d7, d8, d9}, [r3, :64], r8 @ encoding: [0xd8,0x62,0x23,0xf4]


	vld2.8	{d16, d17}, [r0, :64]
	vld2.16	{d16, d17}, [r0, :128]
	vld2.32	{d16, d17}, [r0]
	vld2.8	{d16, d17, d18, d19}, [r0, :64]
	vld2.16	{d16, d17, d18, d19}, [r0, :128]
	vld2.32	{d16, d17, d18, d19}, [r0, :256]

	vld2.8	{d19, d20}, [r0, :64]!
	vld2.16	{d16, d17}, [r0, :128]!
	vld2.32	{q10}, [r0]!
	vld2.8	{d4-d7}, [r0, :64]!
	vld2.16	{d1, d2, d3, d4}, [r0, :128]!
	vld2.32	{q7, q8}, [r0, :256]!

	vld2.8	{d19, d20}, [r0, :64], r6
	vld2.16	{d16, d17}, [r0, :128], r6
	vld2.32	{q10}, [r0], r6
	vld2.8	{d4-d7}, [r0, :64], r6
	vld2.16	{d1, d2, d3, d4}, [r0, :128], r6
	vld2.32	{q7, q8}, [r0, :256], r6

@ CHECK: vld2.8	{d16, d17}, [r0, :64]   @ encoding: [0x1f,0x08,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17}, [r0, :128] @ encoding: [0x6f,0x08,0x60,0xf4]
@ CHECK: vld2.32 {d16, d17}, [r0] @ encoding: [0x8f,0x08,0x60,0xf4]
@ CHECK: vld2.8	{d16, d17, d18, d19}, [r0, :64] @ encoding: [0x1f,0x03,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17, d18, d19}, [r0, :128] @ encoding: [0x6f,0x03,0x60,0xf4]
@ CHECK: vld2.32 {d16, d17, d18, d19}, [r0, :256] @ encoding: [0xbf,0x03,0x60,0xf4]

@ CHECK: vld2.8	{d19, d20}, [r0, :64]!  @ encoding: [0x1d,0x38,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17}, [r0, :128]! @ encoding: [0x6d,0x08,0x60,0xf4]
@ CHECK: vld2.32 {d20, d21}, [r0]!       @ encoding: [0x8d,0x48,0x60,0xf4]
@ CHECK: vld2.8	{d4, d5, d6, d7}, [r0, :64]! @ encoding: [0x1d,0x43,0x20,0xf4]
@ CHECK: vld2.16 {d1, d2, d3, d4}, [r0, :128]! @ encoding: [0x6d,0x13,0x20,0xf4]
@ CHECK: vld2.32 {d14, d15, d16, d17}, [r0, :256]! @ encoding: [0xbd,0xe3,0x20,0xf4]

@ CHECK: vld2.8	{d19, d20}, [r0, :64], r6 @ encoding: [0x16,0x38,0x60,0xf4]
@ CHECK: vld2.16 {d16, d17}, [r0, :128], r6 @ encoding: [0x66,0x08,0x60,0xf4]
@ CHECK: vld2.32 {d20, d21}, [r0], r6    @ encoding: [0x86,0x48,0x60,0xf4]
@ CHECK: vld2.8	{d4, d5, d6, d7}, [r0, :64], r6 @ encoding: [0x16,0x43,0x20,0xf4]
@ CHECK: vld2.16 {d1, d2, d3, d4}, [r0, :128], r6 @ encoding: [0x66,0x13,0x20,0xf4]
@ CHECK: vld2.32 {d14, d15, d16, d17}, [r0, :256], r6 @ encoding: [0xb6,0xe3,0x20,0xf4]


	vld3.8 {d16, d17, d18}, [r1]
	vld3.16 {d6, d7, d8}, [r2]
	vld3.32 {d1, d2, d3}, [r3]
	vld3.8 {d16, d18, d20}, [r0, :64]
	vld3.u16 {d27, d29, d31}, [r4]
	vld3.i32 {d6, d8, d10}, [r5]

	vld3.i8 {d12, d13, d14}, [r6], r1
	vld3.i16 {d11, d12, d13}, [r7], r2
	vld3.u32 {d2, d3, d4}, [r8], r3
	vld3.8 {d4, d6, d8}, [r9], r4
	vld3.u16 {d14, d16, d18}, [r9], r4
	vld3.i32 {d16, d18, d20}, [r10], r5

	vld3.p8 {d6, d7, d8}, [r8]!
	vld3.16 {d9, d10, d11}, [r7]!
	vld3.f32 {d1, d2, d3}, [r6]!
	vld3.8 {d16, d18, d20}, [r0, :64]!
	vld3.p16 {d20, d22, d24}, [r5]!
	vld3.32 {d5, d7, d9}, [r4]!


@ CHECK: vld3.8	{d16, d17, d18}, [r1]   @ encoding: [0x0f,0x04,0x61,0xf4]
@ CHECK: vld3.16	{d6, d7, d8}, [r2]      @ encoding: [0x4f,0x64,0x22,0xf4]
@ CHECK: vld3.32	{d1, d2, d3}, [r3]      @ encoding: [0x8f,0x14,0x23,0xf4]
@ CHECK: vld3.8	{d16, d18, d20}, [r0, :64] @ encoding: [0x1f,0x05,0x60,0xf4]
@ CHECK: vld3.16	{d27, d29, d31}, [r4]   @ encoding: [0x4f,0xb5,0x64,0xf4]
@ CHECK: vld3.32	{d6, d8, d10}, [r5]     @ encoding: [0x8f,0x65,0x25,0xf4]
@ CHECK: vld3.8	{d12, d13, d14}, [r6], r1 @ encoding: [0x01,0xc4,0x26,0xf4]
@ CHECK: vld3.16	{d11, d12, d13}, [r7], r2 @ encoding: [0x42,0xb4,0x27,0xf4]
@ CHECK: vld3.32	{d2, d3, d4}, [r8], r3  @ encoding: [0x83,0x24,0x28,0xf4]
@ CHECK: vld3.8	{d4, d6, d8}, [r9], r4  @ encoding: [0x04,0x45,0x29,0xf4]
@ CHECK: vld3.16	{d14, d16, d18}, [r9], r4 @ encoding: [0x44,0xe5,0x29,0xf4]
@ CHECK: vld3.32	{d16, d18, d20}, [r10], r5 @ encoding: [0x85,0x05,0x6a,0xf4]
@ CHECK: vld3.8	{d6, d7, d8}, [r8]!     @ encoding: [0x0d,0x64,0x28,0xf4]
@ CHECK: vld3.16	{d9, d10, d11}, [r7]!   @ encoding: [0x4d,0x94,0x27,0xf4]
@ CHECK: vld3.32	{d1, d2, d3}, [r6]!     @ encoding: [0x8d,0x14,0x26,0xf4]
@ CHECK: vld3.8	{d16, d18, d20}, [r0, :64]! @ encoding: [0x1d,0x05,0x60,0xf4]
@ CHECK: vld3.16	{d20, d22, d24}, [r5]!  @ encoding: [0x4d,0x45,0x65,0xf4]
@ CHECK: vld3.32	{d5, d7, d9}, [r4]!     @ encoding: [0x8d,0x55,0x24,0xf4]


	vld4.8 {d16, d17, d18, d19}, [r1, :64]
	vld4.16 {d16, d17, d18, d19}, [r2, :128]
	vld4.32 {d16, d17, d18, d19}, [r3, :256]
	vld4.8 {d17, d19, d21, d23}, [r5, :256]
	vld4.16 {d17, d19, d21, d23}, [r7]
	vld4.32 {d16, d18, d20, d22}, [r8]

	vld4.s8 {d16, d17, d18, d19}, [r1, :64]!
	vld4.s16 {d16, d17, d18, d19}, [r2, :128]!
	vld4.s32 {d16, d17, d18, d19}, [r3, :256]!
	vld4.u8 {d17, d19, d21, d23}, [r5, :256]!
	vld4.u16 {d17, d19, d21, d23}, [r7]!
	vld4.u32 {d16, d18, d20, d22}, [r8]!

	vld4.p8 {d16, d17, d18, d19}, [r1, :64], r8
	vld4.p16 {d16, d17, d18, d19}, [r2], r7
	vld4.f32 {d16, d17, d18, d19}, [r3, :64], r5
	vld4.i8 {d16, d18, d20, d22}, [r4, :256], r2
	vld4.i16 {d16, d18, d20, d22}, [r6], r3
	vld4.i32 {d17, d19, d21, d23}, [r9], r4

@ CHECK: vld4.8 {d16, d17, d18, d19}, [r1, :64] @ encoding: [0x1f,0x00,0x61,0xf4]
@ CHECK: vld4.16 {d16, d17, d18, d19}, [r2, :128] @ encoding: [0x6f,0x00,0x62,0xf4]
@ CHECK: vld4.32 {d16, d17, d18, d19}, [r3, :256] @ encoding: [0xbf,0x00,0x63,0xf4]
@ CHECK: vld4.8 {d17, d19, d21, d23}, [r5, :256] @ encoding: [0x3f,0x11,0x65,0xf4]
@ CHECK: vld4.16 {d17, d19, d21, d23}, [r7] @ encoding: [0x4f,0x11,0x67,0xf4]
@ CHECK: vld4.32 {d16, d18, d20, d22}, [r8] @ encoding: [0x8f,0x01,0x68,0xf4]
@ CHECK: vld4.8 {d16, d17, d18, d19}, [r1, :64]! @ encoding: [0x1d,0x00,0x61,0xf4]
@ CHECK: vld4.16 {d16, d17, d18, d19}, [r2, :128]! @ encoding: [0x6d,0x00,0x62,0xf4]
@ CHECK: vld4.32 {d16, d17, d18, d19}, [r3, :256]! @ encoding: [0xbd,0x00,0x63,0xf4]
@ CHECK: vld4.8 {d17, d19, d21, d23}, [r5, :256]! @ encoding: [0x3d,0x11,0x65,0xf4]
@ CHECK: vld4.16 {d17, d19, d21, d23}, [r7]! @ encoding: [0x4d,0x11,0x67,0xf4]
@ CHECK: vld4.32 {d16, d18, d20, d22}, [r8]! @ encoding: [0x8d,0x01,0x68,0xf4]
@ CHECK: vld4.8 {d16, d17, d18, d19}, [r1, :64], r8 @ encoding: [0x18,0x00,0x61,0xf4]
@ CHECK: vld4.16 {d16, d17, d18, d19}, [r2], r7 @ encoding: [0x47,0x00,0x62,0xf4]
@ CHECK: vld4.32 {d16, d17, d18, d19}, [r3, :64], r5 @ encoding: [0x95,0x00,0x63,0xf4]
@ CHECK: vld4.8 {d16, d18, d20, d22}, [r4, :256], r2 @ encoding: [0x32,0x01,0x64,0xf4]
@ CHECK: vld4.16 {d16, d18, d20, d22}, [r6], r3 @ encoding: [0x43,0x01,0x66,0xf4]
@ CHECK: vld4.32 {d17, d19, d21, d23}, [r9], r4 @ encoding: [0x84,0x11,0x69,0xf4]


	vld1.8 {d4[]}, [r1]
	vld1.8 {d4[]}, [r1]!
	vld1.8 {d4[]}, [r1], r3
	vld1.8 {d4[], d5[]}, [r1]
	vld1.8 {d4[], d5[]}, [r1]!
	vld1.8 {d4[], d5[]}, [r1], r3

@ CHECK: vld1.8	{d4[]}, [r1]            @ encoding: [0x0f,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[]}, [r1]!           @ encoding: [0x0d,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[]}, [r1], r3        @ encoding: [0x03,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1]      @ encoding: [0x2f,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1]!     @ encoding: [0x2d,0x4c,0xa1,0xf4]
@ CHECK: vld1.8	{d4[], d5[]}, [r1], r3  @ encoding: [0x23,0x4c,0xa1,0xf4]

	vld1.8	{d16[3]}, [r0]
	vld1.16	{d16[2]}, [r0, :16]
	vld1.32	{d16[1]}, [r0, :32]
        vld1.p8 d12[6], [r2]!
        vld1.i8 d12[6], [r2], r2
        vld1.u16 d12[3], [r2]!
        vld1.16 d12[2], [r2], r2

@ CHECK: vld1.8	{d16[3]}, [r0]          @ encoding: [0x6f,0x00,0xe0,0xf4]
@ CHECK: vld1.16 {d16[2]}, [r0, :16]    @ encoding: [0x9f,0x04,0xe0,0xf4]
@ CHECK: vld1.32 {d16[1]}, [r0, :32]    @ encoding: [0xbf,0x08,0xe0,0xf4]
@ CHECK: vld1.8	{d12[6]}, [r2]!         @ encoding: [0xcd,0xc0,0xa2,0xf4]
@ CHECK: vld1.8	{d12[6]}, [r2], r2      @ encoding: [0xc2,0xc0,0xa2,0xf4]
@ CHECK: vld1.16 {d12[3]}, [r2]!        @ encoding: [0xcd,0xc4,0xa2,0xf4]
@ CHECK: vld1.16 {d12[2]}, [r2], r2     @ encoding: [0x82,0xc4,0xa2,0xf4]


	vld2.8	{d16[1], d17[1]}, [r0, :16]
	vld2.16	{d16[1], d17[1]}, [r0, :32]
	vld2.32	{d16[1], d17[1]}, [r0]
	vld2.16	{d17[1], d19[1]}, [r0]
	vld2.32	{d17[0], d19[0]}, [r0, :64]
	vld2.32	{d17[0], d19[0]}, [r0, :64]!
        vld2.8 {d2[4], d3[4]}, [r2], r3
        vld2.8 {d2[4], d3[4]}, [r2]!
        vld2.8 {d2[4], d3[4]}, [r2]
        vld2.32 {d22[], d23[]}, [r1]
        vld2.32 {d22[], d24[]}, [r1]
        vld2.32 {d10[ ],d11[ ]}, [r3]!
        vld2.32 {d14[ ],d16[ ]}, [r4]!
        vld2.32 {d22[ ],d23[ ]}, [r5], r4
        vld2.32 {d22[ ],d24[ ]}, [r6], r4

@ CHECK: vld2.8	{d16[1], d17[1]}, [r0, :16] @ encoding: [0x3f,0x01,0xe0,0xf4]
@ CHECK: vld2.16 {d16[1], d17[1]}, [r0, :32] @ encoding: [0x5f,0x05,0xe0,0xf4]
@ CHECK: vld2.32 {d16[1], d17[1]}, [r0]  @ encoding: [0x8f,0x09,0xe0,0xf4]
@ CHECK: vld2.16 {d17[1], d19[1]}, [r0]  @ encoding: [0x6f,0x15,0xe0,0xf4]
@ CHECK: vld2.32 {d17[0], d19[0]}, [r0, :64] @ encoding: [0x5f,0x19,0xe0,0xf4]
@ CHECK: vld2.32 {d17[0], d19[0]}, [r0, :64]! @ encoding: [0x5d,0x19,0xe0,0xf4]
@ CHECK: vld2.8	{d2[4], d3[4]}, [r2], r3 @ encoding: [0x83,0x21,0xa2,0xf4]
@ CHECK: vld2.8	{d2[4], d3[4]}, [r2]!   @ encoding: [0x8d,0x21,0xa2,0xf4]
@ CHECK: vld2.8	{d2[4], d3[4]}, [r2]    @ encoding: [0x8f,0x21,0xa2,0xf4]
@ CHECK: vld2.32 {d22[], d23[]}, [r1]    @ encoding: [0x8f,0x6d,0xe1,0xf4]
@ CHECK: vld2.32 {d22[], d24[]}, [r1]    @ encoding: [0xaf,0x6d,0xe1,0xf4]
@ CHECK: vld2.32 {d10[], d11[]}, [r3]!   @ encoding: [0x8d,0xad,0xa3,0xf4]
@ CHECK: vld2.32 {d14[], d16[]}, [r4]!   @ encoding: [0xad,0xed,0xa4,0xf4]
@ CHECK: vld2.32 {d22[], d23[]}, [r5], r4 @ encoding: [0x84,0x6d,0xe5,0xf4]
@ CHECK: vld2.32 {d22[], d24[]}, [r6], r4 @ encoding: [0xa4,0x6d,0xe6,0xf4]


	vld3.8 {d16[1], d17[1], d18[1]}, [r1]
	vld3.16 {d6[1], d7[1], d8[1]}, [r2]
	vld3.32 {d1[1], d2[1], d3[1]}, [r3]
	vld3.u16 {d27[2], d29[2], d31[2]}, [r4]
	vld3.i32 {d6[0], d8[0], d10[0]}, [r5]

	vld3.i8 {d12[3], d13[3], d14[3]}, [r6], r1
	vld3.i16 {d11[2], d12[2], d13[2]}, [r7], r2
	vld3.u32 {d2[1], d3[1], d4[1]}, [r8], r3
	vld3.u16 {d14[2], d16[2], d18[2]}, [r9], r4
	vld3.i32 {d16[0], d18[0], d20[0]}, [r10], r5

	vld3.p8 {d6[6], d7[6], d8[6]}, [r8]!
	vld3.16 {d9[2], d10[2], d11[2]}, [r7]!
	vld3.f32 {d1[1], d2[1], d3[1]}, [r6]!
	vld3.p16 {d20[2], d22[2], d24[2]}, [r5]!
	vld3.32 {d5[0], d7[0], d9[0]}, [r4]!

@ CHECK: vld3.8	{d16[1], d17[1], d18[1]}, [r1] @ encoding: [0x2f,0x02,0xe1,0xf4]
@ CHECK: vld3.16 {d6[1], d7[1], d8[1]}, [r2] @ encoding: [0x4f,0x66,0xa2,0xf4]
@ CHECK: vld3.32 {d1[1], d2[1], d3[1]}, [r3] @ encoding: [0x8f,0x1a,0xa3,0xf4]
@ CHECK: vld3.16 {d27[2], d29[2], d31[2]}, [r4] @ encoding: [0xaf,0xb6,0xe4,0xf4]
@ CHECK: vld3.32 {d6[0], d8[0], d10[0]}, [r5] @ encoding: [0x4f,0x6a,0xa5,0xf4]
@ CHECK: vld3.8	{d12[3], d13[3], d14[3]}, [r6], r1 @ encoding: [0x61,0xc2,0xa6,0xf4]
@ CHECK: vld3.16 {d11[2], d12[2], d13[2]}, [r7], r2 @ encoding: [0x82,0xb6,0xa7,0xf4]
@ CHECK: vld3.32 {d2[1], d3[1], d4[1]}, [r8], r3 @ encoding: [0x83,0x2a,0xa8,0xf4]
@ CHECK: vld3.16 {d14[2], d16[2], d18[2]}, [r9], r4 @ encoding: [0xa4,0xe6,0xa9,0xf4]
@ CHECK: vld3.32 {d16[0], d18[0], d20[0]}, [r10], r5 @ encoding: [0x45,0x0a,0xea,0xf4]
@ CHECK: vld3.8	{d6[6], d7[6], d8[6]}, [r8]! @ encoding: [0xcd,0x62,0xa8,0xf4]
@ CHECK: vld3.16 {d9[2], d10[2], d11[2]}, [r7]! @ encoding: [0x8d,0x96,0xa7,0xf4]
@ CHECK: vld3.32 {d1[1], d2[1], d3[1]}, [r6]! @ encoding: [0x8d,0x1a,0xa6,0xf4]
@ CHECK: vld3.16 {d20[2], d21[2], d22[2]}, [r5]! @ encoding: [0xad,0x46,0xe5,0xf4]
@ CHECK: vld3.32 {d5[0], d7[0], d9[0]}, [r4]! @ encoding: [0x4d,0x5a,0xa4,0xf4]


	vld3.8 {d16[], d17[], d18[]}, [r1]
	vld3.16 {d16[], d17[], d18[]}, [r2]
	vld3.32 {d16[], d17[], d18[]}, [r3]
	vld3.8 {d17[], d19[], d21[]}, [r7]
	vld3.16 {d17[], d19[], d21[]}, [r7]
	vld3.32 {d16[], d18[], d20[]}, [r8]

	vld3.s8 {d16[], d17[], d18[]}, [r1]!
	vld3.s16 {d16[], d17[], d18[]}, [r2]!
	vld3.s32 {d16[], d17[], d18[]}, [r3]!
	vld3.u8 {d17[], d19[], d21[]}, [r7]!
	vld3.u16 {d17[], d19[], d21[]}, [r7]!
	vld3.u32 {d16[], d18[], d20[]}, [r8]!

	vld3.p8 {d16[], d17[], d18[]}, [r1], r8
	vld3.p16 {d16[], d17[], d18[]}, [r2], r7
	vld3.f32 {d16[], d17[], d18[]}, [r3], r5
	vld3.i8 {d16[], d18[], d20[]}, [r6], r3
	vld3.i16 {d16[], d18[], d20[]}, [r6], r3
	vld3.i32 {d17[], d19[], d21[]}, [r9], r4

@ CHECK: vld3.8 {d16[], d17[], d18[]}, [r1] @ encoding: [0x0f,0x0e,0xe1,0xf4]
@ CHECK: vld3.16 {d16[], d17[], d18[]}, [r2] @ encoding: [0x4f,0x0e,0xe2,0xf4]
@ CHECK: vld3.32 {d16[], d17[], d18[]}, [r3] @ encoding: [0x8f,0x0e,0xe3,0xf4]
@ CHECK: vld3.8 {d17[], d19[], d21[]}, [r7] @ encoding: [0x2f,0x1e,0xe7,0xf4]
@ CHECK: vld3.16 {d17[], d19[], d21[]}, [r7] @ encoding: [0x6f,0x1e,0xe7,0xf4]
@ CHECK: vld3.32 {d16[], d18[], d20[]}, [r8] @ encoding: [0xaf,0x0e,0xe8,0xf4]
@ CHECK: vld3.8 {d16[], d17[], d18[]}, [r1]! @ encoding: [0x0d,0x0e,0xe1,0xf4]
@ CHECK: vld3.16 {d16[], d17[], d18[]}, [r2]! @ encoding: [0x4d,0x0e,0xe2,0xf4]
@ CHECK: vld3.32 {d16[], d17[], d18[]}, [r3]! @ encoding: [0x8d,0x0e,0xe3,0xf4]
@ CHECK: vld3.8 {d17[], d18[], d19[]}, [r7]! @ encoding: [0x2d,0x1e,0xe7,0xf4]
@ CHECK: vld3.16 {d17[], d18[], d19[]}, [r7]! @ encoding: [0x6d,0x1e,0xe7,0xf4]
@ CHECK: vld3.32 {d16[], d18[], d20[]}, [r8]! @ encoding: [0xad,0x0e,0xe8,0xf4]
@ CHECK: vld3.8 {d16[], d17[], d18[]}, [r1], r8 @ encoding: [0x08,0x0e,0xe1,0xf4]
@ CHECK: vld3.16 {d16[], d17[], d18[]}, [r2], r7 @ encoding: [0x47,0x0e,0xe2,0xf4]
@ CHECK: vld3.32 {d16[], d17[], d18[]}, [r3], r5 @ encoding: [0x85,0x0e,0xe3,0xf4]
@ CHECK: vld3.8 {d16[], d18[], d20[]}, [r6], r3 @ encoding: [0x23,0x0e,0xe6,0xf4]
@ CHECK: vld3.16 {d16[], d18[], d20[]}, [r6], r3 @ encoding: [0x63,0x0e,0xe6,0xf4]
@ CHECK: vld3.32 {d17[], d19[], d21[]}, [r9], r4 @ encoding: [0xa4,0x1e,0xe9,0xf4]


	vld4.8 {d16[1], d17[1], d18[1], d19[1]}, [r1]
	vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r2]
	vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r3]
	vld4.16 {d17[1], d19[1], d21[1], d23[1]}, [r7]
	vld4.32 {d16[1], d18[1], d20[1], d22[1]}, [r8]

	vld4.s8 {d16[1], d17[1], d18[1], d19[1]}, [r1, :32]!
	vld4.s16 {d16[1], d17[1], d18[1], d19[1]}, [r2, :64]!
	vld4.s32 {d16[1], d17[1], d18[1], d19[1]}, [r3, :128]!
	vld4.u16 {d17[1], d19[1], d21[1], d23[1]}, [r7]!
	vld4.u32 {d16[1], d18[1], d20[1], d22[1]}, [r8]!

	vld4.p8 {d16[1], d17[1], d18[1], d19[1]}, [r1, :32], r8
	vld4.p16 {d16[1], d17[1], d18[1], d19[1]}, [r2], r7
	vld4.f32 {d16[1], d17[1], d18[1], d19[1]}, [r3, :64], r5
	vld4.i16 {d16[1], d18[1], d20[1], d22[1]}, [r6], r3
	vld4.i32 {d17[1], d19[1], d21[1], d23[1]}, [r9], r4

@ CHECK: vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r1] @ encoding: [0x2f,0x03,0xe1,0xf4]
@ CHECK: vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r2] @ encoding: [0x4f,0x07,0xe2,0xf4]
@ CHECK: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r3] @ encoding: [0x8f,0x0b,0xe3,0xf4]
@ CHECK: vld4.16 {d17[1], d19[1], d21[1], d23[1]}, [r7] @ encoding: [0x6f,0x17,0xe7,0xf4]
@ CHECK: vld4.32 {d16[1], d18[1], d20[1], d22[1]}, [r8] @ encoding: [0xcf,0x0b,0xe8,0xf4]
@ CHECK: vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r1, :32]! @ encoding: [0x3d,0x03,0xe1,0xf4]
@ CHECK: vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r2, :64]! @ encoding: [0x5d,0x07,0xe2,0xf4]
@ CHECK: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r3, :128]! @ encoding: [0xad,0x0b,0xe3,0xf4]
@ CHECK: vld4.16 {d17[1], d18[1], d19[1], d20[1]}, [r7]! @ encoding: [0x6d,0x17,0xe7,0xf4]
@ CHECK: vld4.32 {d16[1], d18[1], d20[1], d22[1]}, [r8]! @ encoding: [0xcd,0x0b,0xe8,0xf4]
@ CHECK: vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r1, :32], r8 @ encoding: [0x38,0x03,0xe1,0xf4]
@ CHECK: vld4.16 {d16[1], d17[1], d18[1], d19[1]}, [r2], r7 @ encoding: [0x47,0x07,0xe2,0xf4]
@ CHECK: vld4.32 {d16[1], d17[1], d18[1], d19[1]}, [r3, :64], r5 @ encoding: [0x95,0x0b,0xe3,0xf4]
@ CHECK: vld4.16 {d16[1], d18[1], d20[1], d22[1]}, [r6], r3 @ encoding: [0x63,0x07,0xe6,0xf4]
@ CHECK: vld4.32 {d17[1], d19[1], d21[1], d23[1]}, [r9], r4 @ encoding: [0xc4,0x1b,0xe9,0xf4]


	vld4.8 {d16[], d17[], d18[], d19[]}, [r1]
	vld4.16 {d16[], d17[], d18[], d19[]}, [r2]
	vld4.32 {d16[], d17[], d18[], d19[]}, [r3]
	vld4.8 {d17[], d19[], d21[], d23[]}, [r7]
	vld4.16 {d17[], d19[], d21[], d23[]}, [r7]
	vld4.32 {d16[], d18[], d20[], d22[]}, [r8]

	vld4.s8 {d16[], d17[], d18[], d19[]}, [r1]!
	vld4.s16 {d16[], d17[], d18[], d19[]}, [r2]!
	vld4.s32 {d16[], d17[], d18[], d19[]}, [r3]!
	vld4.u8 {d17[], d19[], d21[], d23[]}, [r7]!
	vld4.u16 {d17[], d19[], d21[], d23[]}, [r7]!
	vld4.u32 {d16[], d18[], d20[], d22[]}, [r8]!

	vld4.p8 {d16[], d17[], d18[], d19[]}, [r1], r8
	vld4.p16 {d16[], d17[], d18[], d19[]}, [r2], r7
	vld4.f32 {d16[], d17[], d18[], d19[]}, [r3], r5
	vld4.i8 {d16[], d18[], d20[], d22[]}, [r6], r3
	vld4.i16 {d16[], d18[], d20[], d22[]}, [r6], r3
	vld4.i32 {d17[], d19[], d21[], d23[]}, [r9], r4

@ CHECK: vld4.8 {d16[], d17[], d18[], d19[]}, [r1] @ encoding: [0x0f,0x0f,0xe1,0xf4]
@ CHECK: vld4.16 {d16[], d17[], d18[], d19[]}, [r2] @ encoding: [0x4f,0x0f,0xe2,0xf4]
@ CHECK: vld4.32 {d16[], d17[], d18[], d19[]}, [r3] @ encoding: [0x8f,0x0f,0xe3,0xf4]
@ CHECK: vld4.8 {d17[], d19[], d21[], d23[]}, [r7] @ encoding: [0x2f,0x1f,0xe7,0xf4]
@ CHECK: vld4.16 {d17[], d19[], d21[], d23[]}, [r7] @ encoding: [0x6f,0x1f,0xe7,0xf4]
@ CHECK: vld4.32 {d16[], d18[], d20[], d22[]}, [r8] @ encoding: [0xaf,0x0f,0xe8,0xf4]
@ CHECK: vld4.8 {d16[], d17[], d18[], d19[]}, [r1]! @ encoding: [0x0d,0x0f,0xe1,0xf4]
@ CHECK: vld4.16 {d16[], d17[], d18[], d19[]}, [r2]! @ encoding: [0x4d,0x0f,0xe2,0xf4]
@ CHECK: vld4.32 {d16[], d17[], d18[], d19[]}, [r3]! @ encoding: [0x8d,0x0f,0xe3,0xf4]
@ CHECK: vld4.8 {d17[], d18[], d19[], d20[]}, [r7]! @ encoding: [0x2d,0x1f,0xe7,0xf4]
@ CHECK: vld4.16 {d17[], d18[], d19[], d20[]}, [r7]! @ encoding: [0x6d,0x1f,0xe7,0xf4]
@ CHECK: vld4.32 {d16[], d18[], d20[], d22[]}, [r8]! @ encoding: [0xad,0x0f,0xe8,0xf4]
@ CHECK: vld4.8 {d16[], d17[], d18[], d19[]}, [r1], r8 @ encoding: [0x08,0x0f,0xe1,0xf4]
@ CHECK: vld4.16 {d16[], d17[], d18[], d19[]}, [r2], r7 @ encoding: [0x47,0x0f,0xe2,0xf4]
@ CHECK: vld4.32 {d16[], d17[], d18[], d19[]}, [r3], r5 @ encoding: [0x85,0x0f,0xe3,0xf4]
@ CHECK: vld4.8 {d16[], d18[], d20[], d22[]}, [r6], r3 @ encoding: [0x23,0x0f,0xe6,0xf4]
@ CHECK: vld4.16 {d16[], d18[], d20[], d22[]}, [r6], r3 @ encoding: [0x63,0x0f,0xe6,0xf4]
@ CHECK: vld4.32 {d17[], d19[], d21[], d23[]}, [r9], r4 @ encoding: [0xa4,0x1f,0xe9,0xf4]

@ Handle 'Q' registers in register lists as if the sub-reg D regs were
@ specified instead.
	vld1.8 {q3}, [r9]
	vld1.8 {q3, q4}, [r9]

@ CHECK: vld1.8	{d6, d7}, [r9]          @ encoding: [0x0f,0x6a,0x29,0xf4]
@ CHECK: vld1.8	{d6, d7, d8, d9}, [r9]  @ encoding: [0x0f,0x62,0x29,0xf4]


@ Spot-check additional size-suffix aliases.
        vld1.8 {d2}, [r2]
        vld1.p8 {d2}, [r2]
        vld1.u8 {d2}, [r2]

        vld1.8 {q2}, [r2]
        vld1.p8 {q2}, [r2]
        vld1.u8 {q2}, [r2]
        vld1.f32 {q2}, [r2]

        vld1.u8 {d2, d3, d4}, [r2]
        vld1.i32 {d2, d3, d4}, [r2]
        vld1.f64 {d2, d3, d4}, [r2]

@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]
@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]
@ CHECK: vld1.8	{d2}, [r2]              @ encoding: [0x0f,0x27,0x22,0xf4]

@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.8	{d4, d5}, [r2]          @ encoding: [0x0f,0x4a,0x22,0xf4]
@ CHECK: vld1.32 {d4, d5}, [r2]         @ encoding: [0x8f,0x4a,0x22,0xf4]

@ CHECK: vld1.8	{d2, d3, d4}, [r2]      @ encoding: [0x0f,0x26,0x22,0xf4]
@ CHECK: vld1.32 {d2, d3, d4}, [r2]     @ encoding: [0x8f,0x26,0x22,0xf4]
@ CHECK: vld1.64 {d2, d3, d4}, [r2]     @ encoding: [0xcf,0x26,0x22,0xf4]


@ Register lists can use the range syntax, just like VLDM
	vld1.f64 {d2-d5}, [r2,:128]!
	vld1.f64 {d2,d3,d4,d5}, [r2,:128]!

@ CHECK: vld1.64 {d2, d3, d4, d5}, [r2, :128]! @ encoding: [0xed,0x22,0x22,0xf4]
@ CHECK: vld1.64 {d2, d3, d4, d5}, [r2, :128]! @ encoding: [0xed,0x22,0x22,0xf4]
