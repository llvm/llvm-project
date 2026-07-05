@ RUN: llvm-mc -triple=armv7-apple-darwin -show-encoding < %s | FileCheck %s
  .syntax unified

@ Shift-by-zero should canonicalize to no shift at all (lsl #0 encoding)
        add r1, r2, r3, lsl #0
        sub r1, r2, r3, ror #0
        eor r1, r2, r3, lsr #0
        orr r1, r2, r3, asr #0
        and r1, r2, r3, ror #0
        bic r1, r2, r3, lsl #0

@ CHECK: add	r1, r2, r3              @ encoding: [0x03,0x10,0x82,0xe0]
@ CHECK: sub	r1, r2, r3              @ encoding: [0x03,0x10,0x42,0xe0]
@ CHECK: eor	r1, r2, r3              @ encoding: [0x03,0x10,0x22,0xe0]
@ CHECK: orr	r1, r2, r3              @ encoding: [0x03,0x10,0x82,0xe1]
@ CHECK: and	r1, r2, r3              @ encoding: [0x03,0x10,0x02,0xe0]
@ CHECK: bic	r1, r2, r3              @ encoding: [0x03,0x10,0xc2,0xe1]
