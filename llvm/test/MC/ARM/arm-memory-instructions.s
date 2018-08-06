@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified
  .globl _func

@ Check that the assembler can handle the documented syntax from the ARM ARM
@ for loads and stores.

_func:
@ CHECK: _func

@------------------------------------------------------------------------------
@ LDR (immediate)
@------------------------------------------------------------------------------
        ldr r5, [r7]
        ldr r6, [r3, #63]
        ldr r2, [r4, #4095]!
        ldr r1, [r2], #30
        ldr r3, [r1], #-30
        ldr r9, [r2], #-0

@ CHECK: ldr	r5, [r7]                @ encoding: [0x00,0x50,0x97,0xe5]
@ CHECK: ldr	r6, [r3, #63]           @ encoding: [0x3f,0x60,0x93,0xe5]
@ CHECK: ldr	r2, [r4, #4095]!        @ encoding: [0xff,0x2f,0xb4,0xe5]
@ CHECK: ldr	r1, [r2], #30           @ encoding: [0x1e,0x10,0x92,0xe4]
@ CHECK: ldr	r3, [r1], #-30          @ encoding: [0x1e,0x30,0x11,0xe4]
@ CHECK: ldr	r9, [r2], #-0           @ encoding: [0x00,0x90,0x12,0xe4]

@------------------------------------------------------------------------------
@ FIXME: LDR (literal)
@------------------------------------------------------------------------------
@ label operands currently assert the show-encoding asm comment helper due
@ to the use of non-contiguous bit ranges for fixups in ARM. Once that's
@ cleaned up, we can write useful assembly testcases for these sorts of
@ instructions.

@------------------------------------------------------------------------------
@ LDR (register)
@------------------------------------------------------------------------------
        ldr r3, [r8, r1]
        ldr r2, [r5, -r3]
        ldr r1, [r5, r9]!
        ldr r6, [r7, -r8]!
        ldr r1, [r0, r2, lsr #3]!
        ldr r5, [r9], r2
        ldr r4, [r3], -r6
        ldr r3, [r8, -r2, lsl #15]
        ldr r1, [r5], r3, asr #15

@ CHECK: ldr	r3, [r8, r1]            @ encoding: [0x01,0x30,0x98,0xe7]
@ CHECK: ldr	r2, [r5, -r3]           @ encoding: [0x03,0x20,0x15,0xe7]
@ CHECK: ldr	r1, [r5, r9]!           @ encoding: [0x09,0x10,0xb5,0xe7]
@ CHECK: ldr	r6, [r7, -r8]!          @ encoding: [0x08,0x60,0x37,0xe7]
@ CHECK: ldr	r1, [r0, r2, lsr #3]!   @ encoding: [0xa2,0x11,0xb0,0xe7]
@ CHECK: ldr	r5, [r9], r2            @ encoding: [0x02,0x50,0x99,0xe6]
@ CHECK: ldr	r4, [r3], -r6           @ encoding: [0x06,0x40,0x13,0xe6]
@ CHECK: ldr	r3, [r8, -r2, lsl #15]  @ encoding: [0x82,0x37,0x18,0xe7]
@ CHECK: ldr	r1, [r5], r3, asr #15   @ encoding: [0xc3,0x17,0x95,0xe6]


@------------------------------------------------------------------------------
@ LDRB (immediate)
@------------------------------------------------------------------------------
        ldrb r3, [r8]
        ldrb r1, [sp, #63]
        ldrb r9, [r3, #4095]!
        ldrb r8, [r1], #22
        ldrb r2, [r7], #-19

@ CHECK: ldrb	r3, [r8]                @ encoding: [0x00,0x30,0xd8,0xe5]
@ CHECK: ldrb	r1, [sp, #63]           @ encoding: [0x3f,0x10,0xdd,0xe5]
@ CHECK: ldrb	r9, [r3, #4095]!        @ encoding: [0xff,0x9f,0xf3,0xe5]
@ CHECK: ldrb	r8, [r1], #22           @ encoding: [0x16,0x80,0xd1,0xe4]
@ CHECK: ldrb	r2, [r7], #-19          @ encoding: [0x13,0x20,0x57,0xe4]


@------------------------------------------------------------------------------
@ LDRB (register)
@------------------------------------------------------------------------------
        ldrb r9, [r8, r5]
        ldrb r1, [r5, -r1]
        ldrb r3, [r5, r2]!
        ldrb r6, [r9, -r3]!
        ldrb r2, [r1], r4
        ldrb r8, [r4], -r5
        ldrb r7, [r12, -r1, lsl #15]
        ldrb r5, [r2], r9, asr #15

@ CHECK: ldrb	r9, [r8, r5]            @ encoding: [0x05,0x90,0xd8,0xe7]
@ CHECK: ldrb	r1, [r5, -r1]           @ encoding: [0x01,0x10,0x55,0xe7]
@ CHECK: ldrb	r3, [r5, r2]!           @ encoding: [0x02,0x30,0xf5,0xe7]
@ CHECK: ldrb	r6, [r9, -r3]!          @ encoding: [0x03,0x60,0x79,0xe7]
@ CHECK: ldrb	r2, [r1], r4            @ encoding: [0x04,0x20,0xd1,0xe6]
@ CHECK: ldrb	r8, [r4], -r5           @ encoding: [0x05,0x80,0x54,0xe6]
@ CHECK: ldrb	r7, [r12, -r1, lsl #15] @ encoding: [0x81,0x77,0x5c,0xe7]
@ CHECK: ldrb	r5, [r2], r9, asr #15   @ encoding: [0xc9,0x57,0xd2,0xe6]


@------------------------------------------------------------------------------
@ LDRBT
@------------------------------------------------------------------------------
@ FIXME: Optional offset operand.
        ldrbt r3, [r1], #4
        ldrbt r2, [r8], #-8
        ldrbt r8, [r7], r6
        ldrbt r1, [r2], -r6, lsl #12


@ CHECK: ldrbt	r3, [r1], #4            @ encoding: [0x04,0x30,0xf1,0xe4]
@ CHECK: ldrbt	r2, [r8], #-8           @ encoding: [0x08,0x20,0x78,0xe4]
@ CHECK: ldrbt	r8, [r7], r6            @ encoding: [0x06,0x80,0xf7,0xe6]
@ CHECK: ldrbt	r1, [r2], -r6, lsl #12  @ encoding: [0x06,0x16,0x72,0xe6]


@------------------------------------------------------------------------------
@ LDRD (immediate)
@------------------------------------------------------------------------------
        ldrd r2, r3, [r5]
        ldrd r6, r7, [r2, #15]
        ldrd r0, r1, [r9, #32]!
        ldrd r6, r7, [r1], #8
        ldrd r0, r1, [r8], #0
        ldrd r0, r1, [r8], #+0
        ldrd r0, r1, [r8], #-0

@ CHECK: ldrd r2, r3, [r5]            @ encoding: [0xd0,0x20,0xc5,0xe1]
@ CHECK: ldrd r6, r7, [r2, #15]       @ encoding: [0xdf,0x60,0xc2,0xe1]
@ CHECK: ldrd r0, r1, [r9, #32]!      @ encoding: [0xd0,0x02,0xe9,0xe1]
@ CHECK: ldrd r6, r7, [r1], #8        @ encoding: [0xd8,0x60,0xc1,0xe0]
@ CHECK: ldrd r0, r1, [r8], #0        @ encoding: [0xd0,0x00,0xc8,0xe0]
@ CHECK: ldrd r0, r1, [r8], #0        @ encoding: [0xd0,0x00,0xc8,0xe0]
@ CHECK: ldrd r0, r1, [r8], #-0       @ encoding: [0xd0,0x00,0x48,0xe0]


@------------------------------------------------------------------------------
@ LDRD (label)
@------------------------------------------------------------------------------
        ldrd r2, r3, Lbaz
Lbaz: .quad 0

@ CHECK: ldrd	r2, r3, Lbaz            @ encoding: [0xd0'A',0x20'A',0x4f'A',0xe1'A']


@------------------------------------------------------------------------------
@ LDRD (register)
@------------------------------------------------------------------------------
        ldrd r4, r5, [r1, r3]
        ldrd r4, r5, [r7, r2]!
        ldrd r0, r1, [r8], r12
        ldrd r0, r1, [r8], -r12

@ CHECK: ldrd r4, r5, [r1, r3]        @ encoding: [0xd3,0x40,0x81,0xe1]
@ CHECK: ldrd r4, r5, [r7, r2]!       @ encoding: [0xd2,0x40,0xa7,0xe1]
@ CHECK: ldrd r0, r1, [r8], r12       @ encoding: [0xdc,0x00,0x88,0xe0]
@ CHECK: ldrd r0, r1, [r8], -r12      @ encoding: [0xdc,0x00,0x08,0xe0]


@------------------------------------------------------------------------------
@ LDRH (immediate)
@------------------------------------------------------------------------------
        ldrh r3, [r4]
        ldrh r2, [r7, #4]
        ldrh r1, [r8, #64]!
        ldrh r12, [sp], #4

@ CHECK: ldrh	r3, [r4]                @ encoding: [0xb0,0x30,0xd4,0xe1]
@ CHECK: ldrh	r2, [r7, #4]            @ encoding: [0xb4,0x20,0xd7,0xe1]
@ CHECK: ldrh	r1, [r8, #64]!          @ encoding: [0xb0,0x14,0xf8,0xe1]
@ CHECK: ldrh	r12, [sp], #4           @ encoding: [0xb4,0xc0,0xdd,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRH (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDRH (register)
@------------------------------------------------------------------------------
        ldrh r6, [r5, r4]
        ldrh r3, [r8, r11]!
        ldrh r1, [r2, -r1]!
        ldrh r9, [r7], r2
        ldrh r4, [r3], -r2

@ CHECK: ldrh	r6, [r5, r4]            @ encoding: [0xb4,0x60,0x95,0xe1]
@ CHECK: ldrh	r3, [r8, r11]!          @ encoding: [0xbb,0x30,0xb8,0xe1]
@ CHECK: ldrh	r1, [r2, -r1]!          @ encoding: [0xb1,0x10,0x32,0xe1]
@ CHECK: ldrh	r9, [r7], r2            @ encoding: [0xb2,0x90,0x97,0xe0]
@ CHECK: ldrh	r4, [r3], -r2           @ encoding: [0xb2,0x40,0x13,0xe0]


@------------------------------------------------------------------------------
@ LDRHT
@------------------------------------------------------------------------------
        ldrht r9, [r7], #128
        ldrht r4, [r3], #-75
        ldrht r9, [r7], r2
        ldrht r4, [r3], -r2

@ CHECK: ldrht	r9, [r7], #128          @ encoding: [0xb0,0x98,0xf7,0xe0]
@ CHECK: ldrht	r4, [r3], #-75          @ encoding: [0xbb,0x44,0x73,0xe0]
@ CHECK: ldrht	r9, [r7], r2            @ encoding: [0xb2,0x90,0xb7,0xe0]
@ CHECK: ldrht	r4, [r3], -r2           @ encoding: [0xb2,0x40,0x33,0xe0]


@------------------------------------------------------------------------------
@ LDRSB (immediate)
@------------------------------------------------------------------------------
        ldrsb r3, [r4]
        ldrsb r2, [r7, #17]
        ldrsb r1, [r8, #255]!
        ldrsb r12, [sp], #9

@ CHECK: ldrsb	r3, [r4]                @ encoding: [0xd0,0x30,0xd4,0xe1]
@ CHECK: ldrsb	r2, [r7, #17]           @ encoding: [0xd1,0x21,0xd7,0xe1]
@ CHECK: ldrsb	r1, [r8, #255]!         @ encoding: [0xdf,0x1f,0xf8,0xe1]
@ CHECK: ldrsb	r12, [sp], #9           @ encoding: [0xd9,0xc0,0xdd,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRSB (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDRSB (register)
@------------------------------------------------------------------------------
        ldrsb r6, [r5, r4]
        ldrsb r3, [r8, r11]!
        ldrsb r1, [r2, -r1]!
        ldrsb r9, [r7], r2
        ldrsb r4, [r3], -r2


@ CHECK: ldrsb	r6, [r5, r4]            @ encoding: [0xd4,0x60,0x95,0xe1]
@ CHECK: ldrsb	r3, [r8, r11]!          @ encoding: [0xdb,0x30,0xb8,0xe1]
@ CHECK: ldrsb	r1, [r2, -r1]!          @ encoding: [0xd1,0x10,0x32,0xe1]
@ CHECK: ldrsb	r9, [r7], r2            @ encoding: [0xd2,0x90,0x97,0xe0]
@ CHECK: ldrsb	r4, [r3], -r2           @ encoding: [0xd2,0x40,0x13,0xe0]


@------------------------------------------------------------------------------
@ LDRSBT
@------------------------------------------------------------------------------
        ldrsbt r5, [r6], #1
        ldrsbt r3, [r8], #-12
        ldrsbt r8, [r9], r5
        ldrsbt r2, [r1], -r4

@ CHECK: ldrsbt	r5, [r6], #1            @ encoding: [0xd1,0x50,0xf6,0xe0]
@ CHECK: ldrsbt	r3, [r8], #-12          @ encoding: [0xdc,0x30,0x78,0xe0]
@ CHECK: ldrsbt	r8, [r9], r5            @ encoding: [0xd5,0x80,0xb9,0xe0]
@ CHECK: ldrsbt	r2, [r1], -r4           @ encoding: [0xd4,0x20,0x31,0xe0]


@------------------------------------------------------------------------------
@ LDRSH (immediate)
@------------------------------------------------------------------------------
        ldrsh r5, [r9]
        ldrsh r4, [r5, #7]
        ldrsh r3, [r6, #55]!
        ldrsh r2, [r7], #-9

@ CHECK: ldrsh	r5, [r9]                @ encoding: [0xf0,0x50,0xd9,0xe1]
@ CHECK: ldrsh	r4, [r5, #7]            @ encoding: [0xf7,0x40,0xd5,0xe1]
@ CHECK: ldrsh	r3, [r6, #55]!          @ encoding: [0xf7,0x33,0xf6,0xe1]
@ CHECK: ldrsh	r2, [r7], #-9           @ encoding: [0xf9,0x20,0x57,0xe0]


@------------------------------------------------------------------------------
@ FIXME: LDRSH (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ LDRSH (register)
@------------------------------------------------------------------------------
        ldrsh r3, [r1, r5]
        ldrsh r4, [r6, r1]!
        ldrsh r5, [r3, -r6]!
        ldrsh r6, [r9], r8
        ldrsh r7, [r8], -r3

@ CHECK: ldrsh	r3, [r1, r5]            @ encoding: [0xf5,0x30,0x91,0xe1]
@ CHECK: ldrsh	r4, [r6, r1]!           @ encoding: [0xf1,0x40,0xb6,0xe1]
@ CHECK: ldrsh	r5, [r3, -r6]!          @ encoding: [0xf6,0x50,0x33,0xe1]
@ CHECK: ldrsh	r6, [r9], r8            @ encoding: [0xf8,0x60,0x99,0xe0]
@ CHECK: ldrsh	r7, [r8], -r3           @ encoding: [0xf3,0x70,0x18,0xe0]


@------------------------------------------------------------------------------
@ LDRSHT
@------------------------------------------------------------------------------
        ldrsht r5, [r6], #1
        ldrsht r3, [r8], #-12
        ldrsht r8, [r9], r5
        ldrsht r2, [r1], -r4

@ CHECK: ldrsht	r5, [r6], #1            @ encoding: [0xf1,0x50,0xf6,0xe0]
@ CHECK: ldrsht	r3, [r8], #-12          @ encoding: [0xfc,0x30,0x78,0xe0]
@ CHECK: ldrsht	r8, [r9], r5            @ encoding: [0xf5,0x80,0xb9,0xe0]
@ CHECK: ldrsht	r2, [r1], -r4           @ encoding: [0xf4,0x20,0x31,0xe0]


@------------------------------------------------------------------------------
@ STR (immediate)
@------------------------------------------------------------------------------
        str r8, [r12]
        str r7, [r1, #12]
        str r3, [r5, #40]!
        str r9, [sp], #4095
        str r1, [r7], #-128
        str r1, [r0], #-0

@ CHECK: str	r8, [r12]               @ encoding: [0x00,0x80,0x8c,0xe5]
@ CHECK: str	r7, [r1, #12]           @ encoding: [0x0c,0x70,0x81,0xe5]
@ CHECK: str	r3, [r5, #40]!          @ encoding: [0x28,0x30,0xa5,0xe5]
@ CHECK: str	r9, [sp], #4095         @ encoding: [0xff,0x9f,0x8d,0xe4]
@ CHECK: str	r1, [r7], #-128         @ encoding: [0x80,0x10,0x07,0xe4]
@ CHECK: str	r1, [r0], #-0           @ encoding: [0x00,0x10,0x00,0xe4]

@------------------------------------------------------------------------------
@ FIXME: STR (literal)
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ STR (register)
@------------------------------------------------------------------------------
        str r9, [r6, r3]
        str r8, [r0, -r2]
        str r7, [r1, r6]!
        str r6, [sp, -r1]!
        str r5, [r3], r9
        str r4, [r2], -r5
        str r3, [r4, -r2, lsl #2]
        str r2, [r7], r3, asr #24

@ CHECK: str	r9, [r6, r3]            @ encoding: [0x03,0x90,0x86,0xe7]
@ CHECK: str	r8, [r0, -r2]           @ encoding: [0x02,0x80,0x00,0xe7]
@ CHECK: str	r7, [r1, r6]!           @ encoding: [0x06,0x70,0xa1,0xe7]
@ CHECK: str	r6, [sp, -r1]!          @ encoding: [0x01,0x60,0x2d,0xe7]
@ CHECK: str	r5, [r3], r9            @ encoding: [0x09,0x50,0x83,0xe6]
@ CHECK: str	r4, [r2], -r5           @ encoding: [0x05,0x40,0x02,0xe6]
@ CHECK: str	r3, [r4, -r2, lsl #2]   @ encoding: [0x02,0x31,0x04,0xe7]
@ CHECK: str	r2, [r7], r3, asr #24   @ encoding: [0x43,0x2c,0x87,0xe6]


@------------------------------------------------------------------------------
@ STRB (immediate)
@------------------------------------------------------------------------------
        strb r9, [r2]
        strb r7, [r1, #3]
        strb r6, [r4, #405]!
        strb r5, [r7], #72
        strb r1, [sp], #-1

@ CHECK: strb	r9, [r2]                @ encoding: [0x00,0x90,0xc2,0xe5]
@ CHECK: strb	r7, [r1, #3]            @ encoding: [0x03,0x70,0xc1,0xe5]
@ CHECK: strb	r6, [r4, #405]!         @ encoding: [0x95,0x61,0xe4,0xe5]
@ CHECK: strb	r5, [r7], #72           @ encoding: [0x48,0x50,0xc7,0xe4]
@ CHECK: strb	r1, [sp], #-1           @ encoding: [0x01,0x10,0x4d,0xe4]

@------------------------------------------------------------------------------
@ FIXME: STRB (literal)
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ STRB (register)
@------------------------------------------------------------------------------
        strb r1, [r2, r9]
        strb r2, [r3, -r8]
        strb r3, [r4, r7]!
        strb r4, [r5, -r6]!
        strb r5, [r6], r5
        strb r6, [r2], -r4
        strb r7, [r12, -r3, lsl #5]
        strb sp, [r7], r2, asr #12

@ CHECK: strb	r1, [r2, r9]            @ encoding: [0x09,0x10,0xc2,0xe7]
@ CHECK: strb	r2, [r3, -r8]           @ encoding: [0x08,0x20,0x43,0xe7]
@ CHECK: strb	r3, [r4, r7]!           @ encoding: [0x07,0x30,0xe4,0xe7]
@ CHECK: strb	r4, [r5, -r6]!          @ encoding: [0x06,0x40,0x65,0xe7]
@ CHECK: strb	r5, [r6], r5            @ encoding: [0x05,0x50,0xc6,0xe6]
@ CHECK: strb	r6, [r2], -r4           @ encoding: [0x04,0x60,0x42,0xe6]
@ CHECK: strb	r7, [r12, -r3, lsl #5]  @ encoding: [0x83,0x72,0x4c,0xe7]
@ CHECK: strb	sp, [r7], r2, asr #12   @ encoding: [0x42,0xd6,0xc7,0xe6]


@------------------------------------------------------------------------------
@ STRBT
@------------------------------------------------------------------------------
@ FIXME: Optional offset operand.
        strbt r6, [r2], #12
        strbt r5, [r6], #-13
        strbt r4, [r9], r5
        strbt r3, [r8], -r2, lsl #3

@ CHECK: strbt	r6, [r2], #12           @ encoding: [0x0c,0x60,0xe2,0xe4]
@ CHECK: strbt	r5, [r6], #-13          @ encoding: [0x0d,0x50,0x66,0xe4]
@ CHECK: strbt	r4, [r9], r5            @ encoding: [0x05,0x40,0xe9,0xe6]
@ CHECK: strbt	r3, [r8], -r2, lsl #3   @ encoding: [0x82,0x31,0x68,0xe6]


@------------------------------------------------------------------------------
@ STRD (immediate)
@------------------------------------------------------------------------------
        strd r2, r3, [r4]
        strd r2, r3, [r6, #1]
        strd r0, r1, [r7, #22]!
        strd r4, r5, [r8], #7
        strd r4, r5, [sp], #0
        strd r6, r7, [lr], #+0
        strd r10, r11, [r9], #-0

@ CHECK: strd	r2, r3, [r4]            @ encoding: [0xf0,0x20,0xc4,0xe1]
@ CHECK: strd	r2, r3, [r6, #1]        @ encoding: [0xf1,0x20,0xc6,0xe1]
@ CHECK: strd	r0, r1, [r7, #22]!      @ encoding: [0xf6,0x01,0xe7,0xe1]
@ CHECK: strd	r4, r5, [r8], #7        @ encoding: [0xf7,0x40,0xc8,0xe0]
@ CHECK: strd	r4, r5, [sp], #0        @ encoding: [0xf0,0x40,0xcd,0xe0]
@ CHECK: strd	r6, r7, [lr], #0        @ encoding: [0xf0,0x60,0xce,0xe0]
@ CHECK: strd	r10, r11, [r9], #-0     @ encoding: [0xf0,0xa0,0x49,0xe0]


@------------------------------------------------------------------------------
@ FIXME: STRD (label)
@------------------------------------------------------------------------------

@------------------------------------------------------------------------------
@ STRD (register)
@------------------------------------------------------------------------------
        strd r8, r9, [r4, r1]
        strd r6, r7, [r3, r9]!
        strd r6, r7, [r5], r8
        strd r4, r5, [r12], -r10

@ CHECK: strd	r8, r9, [r4, r1]        @ encoding: [0xf1,0x80,0x84,0xe1]
@ CHECK: strd	r6, r7, [r3, r9]!       @ encoding: [0xf9,0x60,0xa3,0xe1]
@ CHECK: strd	r6, r7, [r5], r8        @ encoding: [0xf8,0x60,0x85,0xe0]
@ CHECK: strd	r4, r5, [r12], -r10     @ encoding: [0xfa,0x40,0x0c,0xe0]


@------------------------------------------------------------------------------
@ STRH (immediate)
@------------------------------------------------------------------------------
        strh r3, [r4]
        strh r2, [r7, #4]
        strh r1, [r8, #64]!
        strh r12, [sp], #4

@ CHECK: strh	r3, [r4]                @ encoding: [0xb0,0x30,0xc4,0xe1]
@ CHECK: strh	r2, [r7, #4]            @ encoding: [0xb4,0x20,0xc7,0xe1]
@ CHECK: strh	r1, [r8, #64]!          @ encoding: [0xb0,0x14,0xe8,0xe1]
@ CHECK: strh	r12, [sp], #4           @ encoding: [0xb4,0xc0,0xcd,0xe0]


@------------------------------------------------------------------------------
@ FIXME: STRH (label)
@------------------------------------------------------------------------------


@------------------------------------------------------------------------------
@ STRH (register)
@------------------------------------------------------------------------------
        strh r6, [r5, r4]
        strh r3, [r8, r11]!
        strh r1, [r2, -r1]!
        strh r9, [r7], r2
        strh r4, [r3], -r2

@ CHECK: strh	r6, [r5, r4]            @ encoding: [0xb4,0x60,0x85,0xe1]
@ CHECK: strh	r3, [r8, r11]!          @ encoding: [0xbb,0x30,0xa8,0xe1]
@ CHECK: strh	r1, [r2, -r1]!          @ encoding: [0xb1,0x10,0x22,0xe1]
@ CHECK: strh	r9, [r7], r2            @ encoding: [0xb2,0x90,0x87,0xe0]
@ CHECK: strh	r4, [r3], -r2           @ encoding: [0xb2,0x40,0x03,0xe0]

@------------------------------------------------------------------------------
@ STRHT
@------------------------------------------------------------------------------
        strht r2, [r5], #76
        strht r8, [r1], #-25
        strht r5, [r3], r4
        strht r6, [r8], -r0

@ CHECK: strht	r2, [r5], #76           @ encoding: [0xbc,0x24,0xe5,0xe0]
@ CHECK: strht	r8, [r1], #-25          @ encoding: [0xb9,0x81,0x61,0xe0]
@ CHECK: strht	r5, [r3], r4            @ encoding: [0xb4,0x50,0xa3,0xe0]
@ CHECK: strht	r6, [r8], -r0           @ encoding: [0xb0,0x60,0x28,0xe0]

@------------------------------------------------------------------------------
@ GNU Assembler Compatibility
@------------------------------------------------------------------------------

	ldrd r0, [sp]
	strd r0, [sp]

@ CHECK: ldrd r0, r1, [sp]              @ encoding: [0xd0,0x00,0xcd,0xe1]
@ CHECK: strd r0, r1, [sp]              @ encoding: [0xf0,0x00,0xcd,0xe1]

