@ RUN: llvm-mc -mcpu=cortex-a8 -triple thumb-unknown-unknown -show-encoding < %s | FileCheck %s

.code 16

@ CHECK: vld1.8	{d16}, [r0:64]        @ encoding: [0x60,0xf9,0x1f,0x07]
  vld1.8	{d16}, [r0:64]
@ CHECK: vld1.16	{d16}, [r0]             @ encoding: [0x60,0xf9,0x4f,0x07]
  vld1.16	{d16}, [r0]
@ CHECK: vld1.32	{d16}, [r0]             @ encoding: [0x60,0xf9,0x8f,0x07]
  vld1.32	{d16}, [r0]
@ CHECK: vld1.64	{d16}, [r0]             @ encoding: [0x60,0xf9,0xcf,0x07]
  vld1.64	{d16}, [r0]
@ CHECK: vld1.8	{d16, d17}, [r0:64]   @ encoding: [0x60,0xf9,0x1f,0x0a]
  vld1.8	{d16, d17}, [r0:64]
@ CHECK: vld1.16	{d16, d17}, [r0:128]  @ encoding: [0x60,0xf9,0x6f,0x0a]
  vld1.16	{d16, d17}, [r0:128]
@ CHECK: vld1.32	{d16, d17}, [r0]        @ encoding: [0x60,0xf9,0x8f,0x0a]
  vld1.32	{d16, d17}, [r0]
@ CHECK: vld1.64	{d16, d17}, [r0]        @ encoding: [0x60,0xf9,0xcf,0x0a]
  vld1.64	{d16, d17}, [r0]

@ CHECK: vld2.8	{d16, d17}, [r0:64]   @ encoding: [0x60,0xf9,0x1f,0x08]
  vld2.8	{d16, d17}, [r0:64]
@ CHECK: vld2.16	{d16, d17}, [r0:128]  @ encoding: [0x60,0xf9,0x6f,0x08]
  vld2.16	{d16, d17}, [r0:128]
@ CHECK: vld2.32	{d16, d17}, [r0]        @ encoding: [0x60,0xf9,0x8f,0x08]
  vld2.32	{d16, d17}, [r0]
@ CHECK: vld2.8	{d16, d17, d18, d19}, [r0:64] @ encoding: [0x60,0xf9,0x1f,0x03]
  vld2.8	{d16, d17, d18, d19}, [r0:64]
@ CHECK: vld2.16	{d16, d17, d18, d19}, [r0:128] @ encoding: [0x60,0xf9,0x6f,0x03]
  vld2.16	{d16, d17, d18, d19}, [r0:128]
@ CHECK: vld2.32	{d16, d17, d18, d19}, [r0:256] @ encoding: [0x60,0xf9,0xbf,0x03]
  vld2.32	{d16, d17, d18, d19}, [r0:256]

@ CHECK: vld3.8	{d16, d17, d18}, [r0:64] @ encoding: [0x60,0xf9,0x1f,0x04]
  vld3.8	{d16, d17, d18}, [r0:64]
@ CHECK: vld3.16	{d16, d17, d18}, [r0]   @ encoding: [0x60,0xf9,0x4f,0x04]
  vld3.16	{d16, d17, d18}, [r0]
@ CHECK: vld3.32	{d16, d17, d18}, [r0]   @ encoding: [0x60,0xf9,0x8f,0x04]
  vld3.32	{d16, d17, d18}, [r0]
@ CHECK: vld3.8	{d16, d18, d20}, [r0:64]! @ encoding: [0x60,0xf9,0x1d,0x05]
  vld3.8	{d16, d18, d20}, [r0:64]!
@ CHECK: vld3.8	{d17, d19, d21}, [r0:64]! @ encoding: [0x60,0xf9,0x1d,0x15]
  vld3.8	{d17, d19, d21}, [r0:64]!
@ CHECK: vld3.16	{d16, d18, d20}, [r0]!  @ encoding: [0x60,0xf9,0x4d,0x05] 
  vld3.16	{d16, d18, d20}, [r0]!
@ CHECK: vld3.16	{d17, d19, d21}, [r0]!  @ encoding: [0x60,0xf9,0x4d,0x15]
  vld3.16	{d17, d19, d21}, [r0]!
@ CHECK: vld3.32	{d16, d18, d20}, [r0]!  @ encoding: [0x60,0xf9,0x8d,0x05]
  vld3.32	{d16, d18, d20}, [r0]!
@ CHECK: vld3.32	{d17, d19, d21}, [r0]!  @ encoding: [0x60,0xf9,0x8d,0x15]
  vld3.32	{d17, d19, d21}, [r0]!

@ CHECK: vld4.8	{d16, d17, d18, d19}, [r0:64] @ encoding: [0x60,0xf9,0x1f,0x00]
  vld4.8	{d16, d17, d18, d19}, [r0:64]
@ CHECK: vld4.16	{d16, d17, d18, d19}, [r0:128] @ encoding: [0x60,0xf9,0x6f,0x00]
  vld4.16	{d16, d17, d18, d19}, [r0:128]
@ CHECK: vld4.32	{d16, d17, d18, d19}, [r0:256] @ encoding: [0x60,0xf9,0xbf,0x00]
  vld4.32	{d16, d17, d18, d19}, [r0:256]
@ CHECK: vld4.8	{d16, d18, d20, d22}, [r0:256]! @ encoding: [0x60,0xf9,0x3d,0x01]
  vld4.8	{d16, d18, d20, d22}, [r0:256]!
@ CHECK: vld4.8	{d17, d19, d21, d23}, [r0:256]! @ encoding: [0x60,0xf9,0x3d,0x11]
  vld4.8	{d17, d19, d21, d23}, [r0:256]!
@ CHECK: vld4.16	{d16, d18, d20, d22}, [r0]! @ encoding: [0x60,0xf9,0x4d,0x01]
  vld4.16	{d16, d18, d20, d22}, [r0]!
@ CHECK: vld4.16	{d17, d19, d21, d23}, [r0]! @ encoding: [0x60,0xf9,0x4d,0x11]
  vld4.16	{d17, d19, d21, d23}, [r0]!
@ CHECK: vld4.32	{d16, d18, d20, d22}, [r0]! @ encoding: [0x60,0xf9,0x8d,0x01]
  vld4.32	{d16, d18, d20, d22}, [r0]!
@ CHECK: vld4.32	{d17, d19, d21, d23}, [r0]! @ encoding: [0x60,0xf9,0x8d,0x11]
  vld4.32	{d17, d19, d21, d23}, [r0]!

@ CHECK: vld1.8	{d16[3]}, [r0]          @ encoding: [0xe0,0xf9,0x6f,0x00]
  vld1.8	{d16[3]}, [r0]
@ CHECK: vld1.16	{d16[2]}, [r0:16]     @ encoding: [0xe0,0xf9,0x9f,0x04]
  vld1.16	{d16[2]}, [r0:16]
@ CHECK: vld1.32	{d16[1]}, [r0:32]     @ encoding: [0xe0,0xf9,0xbf,0x08]
  vld1.32	{d16[1]}, [r0:32]

@ CHECK: vld2.8	{d16[1], d17[1]}, [r0:16] @ encoding: [0xe0,0xf9,0x3f,0x01]
  vld2.8	{d16[1], d17[1]}, [r0:16]
@ CHECK: vld2.16	{d16[1], d17[1]}, [r0:32] @ encoding: [0xe0,0xf9,0x5f,0x05]
  vld2.16	{d16[1], d17[1]}, [r0:32]
@ CHECK: vld2.32	{d16[1], d17[1]}, [r0]  @ encoding: [0xe0,0xf9,0x8f,0x09]
  vld2.32	{d16[1], d17[1]}, [r0]
@ CHECK: vld2.16	{d17[1], d19[1]}, [r0]  @ encoding: [0xe0,0xf9,0x6f,0x15]
  vld2.16	{d17[1], d19[1]}, [r0]
@ CHECK: vld2.32	{d17[0], d19[0]}, [r0:64] @ encoding: [0xe0,0xf9,0x5f,0x19]
  vld2.32	{d17[0], d19[0]}, [r0:64]

@ CHECK: vld3.8	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xe0,0xf9,0x2f,0x02]
  vld3.8	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vld3.16	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xe0,0xf9,0x4f,0x06]
  vld3.16	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vld3.32	{d16[1], d17[1], d18[1]}, [r0] @ encoding: [0xe0,0xf9,0x8f,0x0a]
  vld3.32	{d16[1], d17[1], d18[1]}, [r0]
@ CHECK: vld3.16	{d16[1], d18[1], d20[1]}, [r0] @ encoding: [0xe0,0xf9,0x6f,0x06]
  vld3.16	{d16[1], d18[1], d20[1]}, [r0]
@ CHECK: vld3.32	{d17[1], d19[1], d21[1]}, [r0] @ encoding: [0xe0,0xf9,0xcf,0x1a]
  vld3.32	{d17[1], d19[1], d21[1]}, [r0]

@ CHECK: vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0:32] @ encoding: [0xe0,0xf9,0x3f,0x03]
  vld4.8	{d16[1], d17[1], d18[1], d19[1]}, [r0:32]
@ CHECK: vld4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0] @ encoding: [0xe0,0xf9,0x4f,0x07]
  vld4.16	{d16[1], d17[1], d18[1], d19[1]}, [r0]
@ CHECK: vld4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0:128] @ encoding: [0xe0,0xf9,0xaf,0x0b]
  vld4.32	{d16[1], d17[1], d18[1], d19[1]}, [r0:128]
@ CHECK: vld4.16	{d16[1], d18[1], d20[1], d22[1]}, [r0:64] @ encoding: [0xe0,0xf9,0x7f,0x07]
  vld4.16	{d16[1], d18[1], d20[1], d22[1]}, [r0:64]
@ CHECK: vld4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0] @ encoding: [0xe0,0xf9,0x4f,0x1b]
  vld4.32	{d17[0], d19[0], d21[0], d23[0]}, [r0]
