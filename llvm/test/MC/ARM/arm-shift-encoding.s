@ RUN: llvm-mc -mcpu=cortex-a8 -triple armv7 -show-encoding < %s | FileCheck %s

	ldr r0, [r0, r0]
	ldr r0, [r0, r0, lsr #32]
	ldr r0, [r0, r0, lsr #16]
	ldr r0, [r0, r0, lsl #0]
	ldr r0, [r0, r0, lsl #16]
	ldr r0, [r0, r0, asr #32]
	ldr r0, [r0, r0, asr #16]
	ldr r0, [r0, r0, rrx]
	ldr r0, [r0, r0, ror #16]

@ CHECK: ldr r0, [r0, r0]          @ encoding: [0x00,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsr #32] @ encoding: [0x20,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsr #16] @ encoding: [0x20,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0]          @ encoding: [0x00,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, lsl #16] @ encoding: [0x00,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, asr #32] @ encoding: [0x40,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, asr #16] @ encoding: [0x40,0x08,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, rrx]     @ encoding: [0x60,0x00,0x90,0xe7]
@ CHECK: ldr r0, [r0, r0, ror #16] @ encoding: [0x60,0x08,0x90,0xe7]

	pld [r0, r0]
	pld [r0, r0, lsr #32]
	pld [r0, r0, lsr #16]
	pld [r0, r0, lsl #0]
	pld [r0, r0, lsl #16]
	pld [r0, r0, asr #32]
	pld [r0, r0, asr #16]
	pld [r0, r0, rrx]
	pld [r0, r0, ror #16]

@ CHECK: [r0, r0]          @ encoding: [0x00,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsr #32] @ encoding: [0x20,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsr #16] @ encoding: [0x20,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0]          @ encoding: [0x00,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, lsl #16] @ encoding: [0x00,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0, asr #32] @ encoding: [0x40,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, asr #16] @ encoding: [0x40,0xf8,0xd0,0xf7]
@ CHECK: [r0, r0, rrx]     @ encoding: [0x60,0xf0,0xd0,0xf7]
@ CHECK: [r0, r0, ror #16] @ encoding: [0x60,0xf8,0xd0,0xf7]

	str r0, [r0, r0]
	str r0, [r0, r0, lsr #32]
	str r0, [r0, r0, lsr #16]
	str r0, [r0, r0, lsl #0]
	str r0, [r0, r0, lsl #16]
	str r0, [r0, r0, asr #32]
	str r0, [r0, r0, asr #16]
	str r0, [r0, r0, rrx]
	str r0, [r0, r0, ror #16]

@ CHECK: str r0, [r0, r0]          @ encoding: [0x00,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsr #32] @ encoding: [0x20,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsr #16] @ encoding: [0x20,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0]          @ encoding: [0x00,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, lsl #16] @ encoding: [0x00,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0, asr #32] @ encoding: [0x40,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, asr #16] @ encoding: [0x40,0x08,0x80,0xe7]
@ CHECK: str r0, [r0, r0, rrx]     @ encoding: [0x60,0x00,0x80,0xe7]
@ CHECK: str r0, [r0, r0, ror #16] @ encoding: [0x60,0x08,0x80,0xe7]

@ Uses printAddrMode2OffsetOperand(), used by LDRBT_POST_IMM LDRBT_POST_REG
@ LDRB_POST_IMM LDRB_POST_REG LDRT_POST_IMM LDRT_POST_REG LDR_POST_IMM
@ LDR_POST_REG STRBT_POST_IMM STRBT_POST_REG STRB_POST_IMM STRB_POST_REG
@ STRT_POST_IMM STRT_POST_REG STR_POST_IMM STR_POST_REG

	ldr r0, [r1], r2, rrx
	ldr r3, [r4], r5, ror #0
	str r6, [r7], r8, lsl #0
	str r9, [r10], r11

@ CHECK: ldr r0, [r1], r2, rrx    @ encoding: [0x62,0x00,0x91,0xe6]
@ CHECK: ldr r3, [r4], r5         @ encoding: [0x05,0x30,0x94,0xe6]
@ CHECK: str r6, [r7], r8         @ encoding: [0x08,0x60,0x87,0xe6]
@ CHECK: str r9, [r10], r11       @ encoding: [0x0b,0x90,0x8a,0xe6]

@ Uses printSORegImmOperand(), used by ADCrsi ADDrsi ANDrsi BICrsi EORrsi
@ ORRrsi RSBrsi RSCrsi SBCrsi SUBrsi CMNzrsi CMPrsi MOVsi MVNsi TEQrsi TSTrsi

	adc sp, lr, pc
	adc r1, r8, r9, lsr #32
	adc r2, r7, pc, lsr #16
	adc r3, r6, r10, lsl #0
	adc r4, r5, lr, lsl #16
	adc r5, r4, r11, asr #32
	adc r6, r3, sp, asr #16
	adc r7, r2, r12, rrx
	adc r8, r1, r0, ror #16

@ CHECK: adc sp, lr, pc           @ encoding: [0x0f,0xd0,0xae,0xe0]
@ CHECK: adc r1, r8, r9, lsr #32  @ encoding: [0x29,0x10,0xa8,0xe0]
@ CHECK: adc r2, r7, pc, lsr #16  @ encoding: [0x2f,0x28,0xa7,0xe0]
@ CHECK: adc r3, r6, r10          @ encoding: [0x0a,0x30,0xa6,0xe0]
@ CHECK: adc r4, r5, lr, lsl #16  @ encoding: [0x0e,0x48,0xa5,0xe0]
@ CHECK: adc r5, r4, r11, asr #32 @ encoding: [0x4b,0x50,0xa4,0xe0]
@ CHECK: adc r6, r3, sp, asr #16  @ encoding: [0x4d,0x68,0xa3,0xe0]
@ CHECK: adc r7, r2, r12, rrx     @ encoding: [0x6c,0x70,0xa2,0xe0]
@ CHECK: adc r8, r1, r0, ror #16  @ encoding: [0x60,0x88,0xa1,0xe0]

	cmp sp, lr
	cmp r1, r8, lsr #32
	cmp r2, r7, lsr #16
	cmp r3, r6, lsl #0
	cmp r4, r5, lsl #16
	cmp r5, r4, asr #32
	cmp r6, r3, asr #16
	cmp r7, r2, rrx
	cmp r8, r1, ror #16

@ CHECK: cmp sp, lr           @ encoding: [0x0e,0x00,0x5d,0xe1]
@ CHECK: cmp r1, r8, lsr #32  @ encoding: [0x28,0x00,0x51,0xe1]
@ CHECK: cmp r2, r7, lsr #16  @ encoding: [0x27,0x08,0x52,0xe1]
@ CHECK: cmp r3, r6           @ encoding: [0x06,0x00,0x53,0xe1]
@ CHECK: cmp r4, r5, lsl #16  @ encoding: [0x05,0x08,0x54,0xe1]
@ CHECK: cmp r5, r4, asr #32  @ encoding: [0x44,0x00,0x55,0xe1]
@ CHECK: cmp r6, r3, asr #16  @ encoding: [0x43,0x08,0x56,0xe1]
@ CHECK: cmp r7, r2, rrx      @ encoding: [0x62,0x00,0x57,0xe1]
@ CHECK: cmp r8, r1, ror #16  @ encoding: [0x61,0x08,0x58,0xe1]
