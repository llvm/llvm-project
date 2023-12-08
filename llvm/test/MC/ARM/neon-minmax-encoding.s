@ RUN: llvm-mc -mcpu=cortex-a8 -triple arm-unknown-unknown -show-encoding < %s | FileCheck %s

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

@ CHECK: vmax.s8	d1, d2, d3      @ encoding: [0x03,0x16,0x02,0xf2]
@ CHECK: vmax.s16	d4, d5, d6      @ encoding: [0x06,0x46,0x15,0xf2]
@ CHECK: vmax.s32	d7, d8, d9      @ encoding: [0x09,0x76,0x28,0xf2]
@ CHECK: vmax.u8	d10, d11, d12   @ encoding: [0x0c,0xa6,0x0b,0xf3]
@ CHECK: vmax.u16	d13, d14, d15   @ encoding: [0x0f,0xd6,0x1e,0xf3]
@ CHECK: vmax.u32	d16, d17, d18   @ encoding: [0xa2,0x06,0x61,0xf3]
@ CHECK: vmax.f32	d19, d20, d21   @ encoding: [0xa5,0x3f,0x44,0xf2]
@ CHECK: vmax.s8	d2, d2, d3      @ encoding: [0x03,0x26,0x02,0xf2]
@ CHECK: vmax.s16	d5, d5, d6      @ encoding: [0x06,0x56,0x15,0xf2]
@ CHECK: vmax.s32	d8, d8, d9      @ encoding: [0x09,0x86,0x28,0xf2]
@ CHECK: vmax.u8	d11, d11, d12   @ encoding: [0x0c,0xb6,0x0b,0xf3]
@ CHECK: vmax.u16	d14, d14, d15   @ encoding: [0x0f,0xe6,0x1e,0xf3]
@ CHECK: vmax.u32	d17, d17, d18   @ encoding: [0xa2,0x16,0x61,0xf3]
@ CHECK: vmax.f32	d20, d20, d21   @ encoding: [0xa5,0x4f,0x44,0xf2]
@ CHECK: vmax.s8	q1, q2, q3      @ encoding: [0x46,0x26,0x04,0xf2]
@ CHECK: vmax.s16	q4, q5, q6      @ encoding: [0x4c,0x86,0x1a,0xf2]
@ CHECK: vmax.s32	q7, q8, q9      @ encoding: [0xe2,0xe6,0x20,0xf2]
@ CHECK: vmax.u8	q10, q11, q12   @ encoding: [0xe8,0x46,0x46,0xf3]
@ CHECK: vmax.u16	q13, q14, q15   @ encoding: [0xee,0xa6,0x5c,0xf3]
@ CHECK: vmax.u32	q6, q7, q8      @ encoding: [0x60,0xc6,0x2e,0xf3]
@ CHECK: vmax.f32	q9, q5, q1      @ encoding: [0x42,0x2f,0x4a,0xf2]
@ CHECK: vmax.s8	q2, q2, q3      @ encoding: [0x46,0x46,0x04,0xf2]
@ CHECK: vmax.s16	q5, q5, q6      @ encoding: [0x4c,0xa6,0x1a,0xf2]
@ CHECK: vmax.s32	q8, q8, q9      @ encoding: [0xe2,0x06,0x60,0xf2]
@ CHECK: vmax.u8	q11, q11, q2    @ encoding: [0xc4,0x66,0x46,0xf3]
@ CHECK: vmax.u16	q4, q4, q5      @ encoding: [0x4a,0x86,0x18,0xf3]
@ CHECK: vmax.u32	q7, q7, q8      @ encoding: [0x60,0xe6,0x2e,0xf3]
@ CHECK: vmax.f32	q2, q2, q1      @ encoding: [0x42,0x4f,0x04,0xf2]


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

@ CHECK: vmin.s8	d1, d2, d3      @ encoding: [0x13,0x16,0x02,0xf2]
@ CHECK: vmin.s16	d4, d5, d6      @ encoding: [0x16,0x46,0x15,0xf2]
@ CHECK: vmin.s32	d7, d8, d9      @ encoding: [0x19,0x76,0x28,0xf2]
@ CHECK: vmin.u8	d10, d11, d12   @ encoding: [0x1c,0xa6,0x0b,0xf3]
@ CHECK: vmin.u16	d13, d14, d15   @ encoding: [0x1f,0xd6,0x1e,0xf3]
@ CHECK: vmin.u32	d16, d17, d18   @ encoding: [0xb2,0x06,0x61,0xf3]
@ CHECK: vmin.f32	d19, d20, d21   @ encoding: [0xa5,0x3f,0x64,0xf2]
@ CHECK: vmin.s8	d2, d2, d3      @ encoding: [0x13,0x26,0x02,0xf2]
@ CHECK: vmin.s16	d5, d5, d6      @ encoding: [0x16,0x56,0x15,0xf2]
@ CHECK: vmin.s32	d8, d8, d9      @ encoding: [0x19,0x86,0x28,0xf2]
@ CHECK: vmin.u8	d11, d11, d12   @ encoding: [0x1c,0xb6,0x0b,0xf3]
@ CHECK: vmin.u16	d14, d14, d15   @ encoding: [0x1f,0xe6,0x1e,0xf3]
@ CHECK: vmin.u32	d17, d17, d18   @ encoding: [0xb2,0x16,0x61,0xf3]
@ CHECK: vmin.f32	d20, d20, d21   @ encoding: [0xa5,0x4f,0x64,0xf2]
@ CHECK: vmin.s8	q1, q2, q3      @ encoding: [0x56,0x26,0x04,0xf2]
@ CHECK: vmin.s16	q4, q5, q6      @ encoding: [0x5c,0x86,0x1a,0xf2]
@ CHECK: vmin.s32	q7, q8, q9      @ encoding: [0xf2,0xe6,0x20,0xf2]
@ CHECK: vmin.u8	q10, q11, q12   @ encoding: [0xf8,0x46,0x46,0xf3]
@ CHECK: vmin.u16	q13, q14, q15   @ encoding: [0xfe,0xa6,0x5c,0xf3]
@ CHECK: vmin.u32	q6, q7, q8      @ encoding: [0x70,0xc6,0x2e,0xf3]
@ CHECK: vmin.f32	q9, q5, q1      @ encoding: [0x42,0x2f,0x6a,0xf2]
@ CHECK: vmin.s8	q2, q2, q3      @ encoding: [0x56,0x46,0x04,0xf2]
@ CHECK: vmin.s16	q5, q5, q6      @ encoding: [0x5c,0xa6,0x1a,0xf2]
@ CHECK: vmin.s32	q8, q8, q9      @ encoding: [0xf2,0x06,0x60,0xf2]
@ CHECK: vmin.u8	q11, q11, q2    @ encoding: [0xd4,0x66,0x46,0xf3]
@ CHECK: vmin.u16	q4, q4, q5      @ encoding: [0x5a,0x86,0x18,0xf3]
@ CHECK: vmin.u32	q7, q7, q8      @ encoding: [0x70,0xe6,0x2e,0xf3]
@ CHECK: vmin.f32	q2, q2, q1      @ encoding: [0x42,0x4f,0x24,0xf2]
