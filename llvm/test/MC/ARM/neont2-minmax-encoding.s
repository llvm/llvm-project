@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

        vmax.s8 d1, d2, d3
        vmax.s16 d4, d5, d6
        vmax.s32 d7, d8, d9
        vmax.u8 d10, d11, d12
        vmax.u16 d13, d14, d15
        vmax.u32 d16, d17, d18
        vmax.f32 d19, d20, d21

        vmax.s8 d2, d3
        vmax.s16 d5, d6
        vmax.s32 d8, d9
        vmax.u8 d11, d12
        vmax.u16 d14, d15
        vmax.u32 d17, d18
        vmax.f32 d20, d21

        vmax.s8 q1, q2, q3
        vmax.s16 q4, q5, q6
        vmax.s32 q7, q8, q9
        vmax.u8 q10, q11, q12
        vmax.u16 q13, q14, q15
        vmax.u32 q6, q7, q8
        vmax.f32 q9, q5, q1

        vmax.s8 q2, q3
        vmax.s16 q5, q6
        vmax.s32 q8, q9
        vmax.u8 q11, q2
        vmax.u16 q4, q5
        vmax.u32 q7, q8
        vmax.f32 q2, q1

@ CHECK: vmax.s8	d1, d2, d3      @ encoding: [0x02,0xef,0x03,0x16]
@ CHECK: vmax.s16	d4, d5, d6      @ encoding: [0x15,0xef,0x06,0x46]
@ CHECK: vmax.s32	d7, d8, d9      @ encoding: [0x28,0xef,0x09,0x76]
@ CHECK: vmax.u8	d10, d11, d12   @ encoding: [0x0b,0xff,0x0c,0xa6]
@ CHECK: vmax.u16	d13, d14, d15   @ encoding: [0x1e,0xff,0x0f,0xd6]
@ CHECK: vmax.u32	d16, d17, d18   @ encoding: [0x61,0xff,0xa2,0x06]
@ CHECK: vmax.f32	d19, d20, d21   @ encoding: [0x44,0xef,0xa5,0x3f]
@ CHECK: vmax.s8	d2, d2, d3      @ encoding: [0x02,0xef,0x03,0x26]
@ CHECK: vmax.s16	d5, d5, d6      @ encoding: [0x15,0xef,0x06,0x56]
@ CHECK: vmax.s32	d8, d8, d9      @ encoding: [0x28,0xef,0x09,0x86]
@ CHECK: vmax.u8	d11, d11, d12   @ encoding: [0x0b,0xff,0x0c,0xb6]
@ CHECK: vmax.u16	d14, d14, d15   @ encoding: [0x1e,0xff,0x0f,0xe6]
@ CHECK: vmax.u32	d17, d17, d18   @ encoding: [0x61,0xff,0xa2,0x16]
@ CHECK: vmax.f32	d20, d20, d21   @ encoding: [0x44,0xef,0xa5,0x4f]
@ CHECK: vmax.s8	q1, q2, q3      @ encoding: [0x04,0xef,0x46,0x26]
@ CHECK: vmax.s16	q4, q5, q6      @ encoding: [0x1a,0xef,0x4c,0x86]
@ CHECK: vmax.s32	q7, q8, q9      @ encoding: [0x20,0xef,0xe2,0xe6]
@ CHECK: vmax.u8	q10, q11, q12   @ encoding: [0x46,0xff,0xe8,0x46]
@ CHECK: vmax.u16	q13, q14, q15   @ encoding: [0x5c,0xff,0xee,0xa6]
@ CHECK: vmax.u32	q6, q7, q8      @ encoding: [0x2e,0xff,0x60,0xc6]
@ CHECK: vmax.f32	q9, q5, q1      @ encoding: [0x4a,0xef,0x42,0x2f]
@ CHECK: vmax.s8	q2, q2, q3      @ encoding: [0x04,0xef,0x46,0x46]
@ CHECK: vmax.s16	q5, q5, q6      @ encoding: [0x1a,0xef,0x4c,0xa6]
@ CHECK: vmax.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xe2,0x06]
@ CHECK: vmax.u8	q11, q11, q2    @ encoding: [0x46,0xff,0xc4,0x66]
@ CHECK: vmax.u16	q4, q4, q5      @ encoding: [0x18,0xff,0x4a,0x86]
@ CHECK: vmax.u32	q7, q7, q8      @ encoding: [0x2e,0xff,0x60,0xe6]
@ CHECK: vmax.f32	q2, q2, q1      @ encoding: [0x04,0xef,0x42,0x4f]


        vmin.s8 d1, d2, d3
        vmin.s16 d4, d5, d6
        vmin.s32 d7, d8, d9
        vmin.u8 d10, d11, d12
        vmin.u16 d13, d14, d15
        vmin.u32 d16, d17, d18
        vmin.f32 d19, d20, d21

        vmin.s8 d2, d3
        vmin.s16 d5, d6
        vmin.s32 d8, d9
        vmin.u8 d11, d12
        vmin.u16 d14, d15
        vmin.u32 d17, d18
        vmin.f32 d20, d21

        vmin.s8 q1, q2, q3
        vmin.s16 q4, q5, q6
        vmin.s32 q7, q8, q9
        vmin.u8 q10, q11, q12
        vmin.u16 q13, q14, q15
        vmin.u32 q6, q7, q8
        vmin.f32 q9, q5, q1

        vmin.s8 q2, q3
        vmin.s16 q5, q6
        vmin.s32 q8, q9
        vmin.u8 q11, q2
        vmin.u16 q4, q5
        vmin.u32 q7, q8
        vmin.f32 q2, q1

@ CHECK: vmin.s8	d1, d2, d3      @ encoding: [0x02,0xef,0x13,0x16]
@ CHECK: vmin.s16	d4, d5, d6      @ encoding: [0x15,0xef,0x16,0x46]
@ CHECK: vmin.s32	d7, d8, d9      @ encoding: [0x28,0xef,0x19,0x76]
@ CHECK: vmin.u8	d10, d11, d12   @ encoding: [0x0b,0xff,0x1c,0xa6]
@ CHECK: vmin.u16	d13, d14, d15   @ encoding: [0x1e,0xff,0x1f,0xd6]
@ CHECK: vmin.u32	d16, d17, d18   @ encoding: [0x61,0xff,0xb2,0x06]
@ CHECK: vmin.f32	d19, d20, d21   @ encoding: [0x64,0xef,0xa5,0x3f]
@ CHECK: vmin.s8	d2, d2, d3      @ encoding: [0x02,0xef,0x13,0x26]
@ CHECK: vmin.s16	d5, d5, d6      @ encoding: [0x15,0xef,0x16,0x56]
@ CHECK: vmin.s32	d8, d8, d9      @ encoding: [0x28,0xef,0x19,0x86]
@ CHECK: vmin.u8	d11, d11, d12   @ encoding: [0x0b,0xff,0x1c,0xb6]
@ CHECK: vmin.u16	d14, d14, d15   @ encoding: [0x1e,0xff,0x1f,0xe6]
@ CHECK: vmin.u32	d17, d17, d18   @ encoding: [0x61,0xff,0xb2,0x16]
@ CHECK: vmin.f32	d20, d20, d21   @ encoding: [0x64,0xef,0xa5,0x4f]
@ CHECK: vmin.s8	q1, q2, q3      @ encoding: [0x04,0xef,0x56,0x26]
@ CHECK: vmin.s16	q4, q5, q6      @ encoding: [0x1a,0xef,0x5c,0x86]
@ CHECK: vmin.s32	q7, q8, q9      @ encoding: [0x20,0xef,0xf2,0xe6]
@ CHECK: vmin.u8	q10, q11, q12   @ encoding: [0x46,0xff,0xf8,0x46]
@ CHECK: vmin.u16	q13, q14, q15   @ encoding: [0x5c,0xff,0xfe,0xa6]
@ CHECK: vmin.u32	q6, q7, q8      @ encoding: [0x2e,0xff,0x70,0xc6]
@ CHECK: vmin.f32	q9, q5, q1      @ encoding: [0x6a,0xef,0x42,0x2f]
@ CHECK: vmin.s8	q2, q2, q3      @ encoding: [0x04,0xef,0x56,0x46]
@ CHECK: vmin.s16	q5, q5, q6      @ encoding: [0x1a,0xef,0x5c,0xa6]
@ CHECK: vmin.s32	q8, q8, q9      @ encoding: [0x60,0xef,0xf2,0x06]
@ CHECK: vmin.u8	q11, q11, q2    @ encoding: [0x46,0xff,0xd4,0x66]
@ CHECK: vmin.u16	q4, q4, q5      @ encoding: [0x18,0xff,0x5a,0x86]
@ CHECK: vmin.u32	q7, q7, q8      @ encoding: [0x2e,0xff,0x70,0xe6]
@ CHECK: vmin.f32	q2, q2, q1      @ encoding: [0x24,0xef,0x42,0x4f]
