@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumbv7 -show-encoding < %s | FileCheck %s

@ Uses printT2SOOperand(), used by t2ADCrs t2ADDrs t2ANDrs t2BICrs t2EORrs
@ t2ORNrs t2ORRrs t2RSBrs t2SBCrs t2SUBrs t2CMNzrs t2CMPrs t2MOVSsi t2MOVsi
@ t2MVNs t2TEQrs t2TSTrs

	sbc.w r12, lr, r0
	sbc.w r1, r8, r9, lsr #32
	sbc.w r2, r7, pc, lsr #16
	sbc.w r3, r6, r10, lsl #0
	sbc.w r4, r5, lr, lsl #16
	sbc.w r5, r4, r11, asr #32
	sbc.w r6, r3, sp, asr #16
	sbc.w r7, r2, r12, rrx
	sbc.w r8, r1, r0, ror #16

@ CHECK: sbc.w r12, lr, r0          @ encoding: [0x6e,0xeb,0x00,0x0c]
@ CHECK: sbc.w r1, r8, r9, lsr #32  @ encoding: [0x68,0xeb,0x19,0x01]
@ CHECK: sbc.w r2, r7, pc, lsr #16  @ encoding: [0x67,0xeb,0x1f,0x42]
@ CHECK: sbc.w r3, r6, r10          @ encoding: [0x66,0xeb,0x0a,0x03]
@ CHECK: sbc.w r4, r5, lr, lsl #16  @ encoding: [0x65,0xeb,0x0e,0x44]
@ CHECK: sbc.w r5, r4, r11, asr #32 @ encoding: [0x64,0xeb,0x2b,0x05]
@ CHECK: sbc.w r6, r3, sp, asr #16  @ encoding: [0x63,0xeb,0x2d,0x46]
@ CHECK: sbc.w r7, r2, r12, rrx     @ encoding: [0x62,0xeb,0x3c,0x07]
@ CHECK: sbc.w r8, r1, r0, ror #16  @ encoding: [0x61,0xeb,0x30,0x48]

	and.w r12, lr, r0
	and.w r1, r8, r9, lsr #32
	and.w r2, r7, pc, lsr #16
	and.w r3, r6, r10, lsl #0
	and.w r4, r5, lr, lsl #16
	and.w r5, r4, r11, asr #32
	and.w r6, r3, sp, asr #16
	and.w r7, r2, r12, rrx
	and.w r8, r1, r0, ror #16

@ CHECK: and.w r12, lr, r0          @ encoding: [0x0e,0xea,0x00,0x0c]
@ CHECK: and.w r1, r8, r9, lsr #32  @ encoding: [0x08,0xea,0x19,0x01]
@ CHECK: and.w r2, r7, pc, lsr #16  @ encoding: [0x07,0xea,0x1f,0x42]
@ CHECK: and.w r3, r6, r10          @ encoding: [0x06,0xea,0x0a,0x03]
@ CHECK: and.w r4, r5, lr, lsl #16  @ encoding: [0x05,0xea,0x0e,0x44]
@ CHECK: and.w r5, r4, r11, asr #32 @ encoding: [0x04,0xea,0x2b,0x05]
@ CHECK: and.w r6, r3, sp, asr #16  @ encoding: [0x03,0xea,0x2d,0x46]
@ CHECK: and.w r7, r2, r12, rrx     @ encoding: [0x02,0xea,0x3c,0x07]
@ CHECK: and.w r8, r1, r0, ror #16  @ encoding: [0x01,0xea,0x30,0x48]
