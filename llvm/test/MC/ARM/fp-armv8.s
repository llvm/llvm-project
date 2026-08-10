@ RUN: llvm-mc -triple armv8 -mattr=+fp-armv8 -show-encoding < %s | FileCheck %s

@ VCVT{B,T}

  vcvtt.f64.f16 d3, s1
@ CHECK: vcvtt.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xee]
  vcvtt.f16.f64 s5, d12
@ CHECK: vcvtt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xee]

  vcvtb.f64.f16 d3, s1
@ CHECK: vcvtb.f64.f16 d3, s1     @ encoding: [0x60,0x3b,0xb2,0xee]
  vcvtb.f16.f64 s4, d1
@ CHECK: vcvtb.f16.f64 s4, d1     @ encoding: [0x41,0x2b,0xb3,0xee]

  vcvttge.f64.f16 d3, s1
@ CHECK: vcvttge.f64.f16 d3, s1      @ encoding: [0xe0,0x3b,0xb2,0xae]
  vcvttgt.f16.f64 s5, d12
@ CHECK: vcvttgt.f16.f64 s5, d12     @ encoding: [0xcc,0x2b,0xf3,0xce]

  vcvtbeq.f64.f16 d3, s1
@ CHECK: vcvtbeq.f64.f16 d3, s1     @ encoding: [0x60,0x3b,0xb2,0x0e]
  vcvtblt.f16.f64 s4, d1
@ CHECK: vcvtblt.f16.f64 s4, d1     @ encoding: [0x41,0x2b,0xb3,0xbe]


@ VCVT{A,N,P,M}

  vcvta.s32.f32 s2, s3
@ CHECK: vcvta.s32.f32 s2, s3     @ encoding: [0xe1,0x1a,0xbc,0xfe]
  vcvta.s32.f64 s2, d3
@ CHECK: vcvta.s32.f64 s2, d3     @ encoding: [0xc3,0x1b,0xbc,0xfe]
  vcvtn.s32.f32 s6, s23
@ CHECK: vcvtn.s32.f32 s6, s23     @ encoding: [0xeb,0x3a,0xbd,0xfe]
  vcvtn.s32.f64 s6, d23
@ CHECK: vcvtn.s32.f64 s6, d23     @ encoding: [0xe7,0x3b,0xbd,0xfe]
  vcvtp.s32.f32 s0, s4
@ CHECK: vcvtp.s32.f32 s0, s4     @ encoding: [0xc2,0x0a,0xbe,0xfe]
  vcvtp.s32.f64 s0, d4
@ CHECK: vcvtp.s32.f64 s0, d4     @ encoding: [0xc4,0x0b,0xbe,0xfe]
  vcvtm.s32.f32 s17, s8
@ CHECK: vcvtm.s32.f32 s17, s8     @ encoding: [0xc4,0x8a,0xff,0xfe]
  vcvtm.s32.f64 s17, d8
@ CHECK: vcvtm.s32.f64 s17, d8     @ encoding: [0xc8,0x8b,0xff,0xfe]

  vcvta.u32.f32 s2, s3
@ CHECK: vcvta.u32.f32 s2, s3     @ encoding: [0x61,0x1a,0xbc,0xfe]
  vcvta.u32.f64 s2, d3
@ CHECK: vcvta.u32.f64 s2, d3     @ encoding: [0x43,0x1b,0xbc,0xfe]
  vcvtn.u32.f32 s6, s23
@ CHECK: vcvtn.u32.f32 s6, s23     @ encoding: [0x6b,0x3a,0xbd,0xfe]
  vcvtn.u32.f64 s6, d23
@ CHECK: vcvtn.u32.f64 s6, d23     @ encoding: [0x67,0x3b,0xbd,0xfe]
  vcvtp.u32.f32 s0, s4
@ CHECK: vcvtp.u32.f32 s0, s4     @ encoding: [0x42,0x0a,0xbe,0xfe]
  vcvtp.u32.f64 s0, d4
@ CHECK: vcvtp.u32.f64 s0, d4     @ encoding: [0x44,0x0b,0xbe,0xfe]
  vcvtm.u32.f32 s17, s8
@ CHECK: vcvtm.u32.f32 s17, s8     @ encoding: [0x44,0x8a,0xff,0xfe]
  vcvtm.u32.f64 s17, d8
@ CHECK: vcvtm.u32.f64 s17, d8     @ encoding: [0x48,0x8b,0xff,0xfe]


@ VSEL
  vselge.f32 s4, s1, s23
@ CHECK: vselge.f32 s4, s1, s23    @ encoding: [0xab,0x2a,0x20,0xfe]
  vselge.f64 d30, d31, d23
@ CHECK: vselge.f64 d30, d31, d23  @ encoding: [0xa7,0xeb,0x6f,0xfe]
  vselgt.f32 s0, s1, s0
@ CHECK: vselgt.f32 s0, s1, s0    @ encoding: [0x80,0x0a,0x30,0xfe]
  vselgt.f64 d5, d10, d20
@ CHECK: vselgt.f64 d5, d10, d20  @ encoding: [0x24,0x5b,0x3a,0xfe]
  vseleq.f32 s30, s28, s23
@ CHECK: vseleq.f32 s30, s28, s23 @ encoding: [0x2b,0xfa,0x0e,0xfe]
  vseleq.f64 d2, d4, d8
@ CHECK: vseleq.f64 d2, d4, d8    @ encoding: [0x08,0x2b,0x04,0xfe]
  vselvs.f32 s21, s16, s14
@ CHECK: vselvs.f32 s21, s16, s14 @ encoding: [0x07,0xaa,0x58,0xfe]
  vselvs.f64 d0, d1, d31
@ CHECK: vselvs.f64 d0, d1, d31   @ encoding: [0x2f,0x0b,0x11,0xfe]


@ VMAXNM / VMINNM
  vmaxnm.f32 s5, s12, s0
@ CHECK: vmaxnm.f32 s5, s12, s0    @ encoding: [0x00,0x2a,0xc6,0xfe]
  vmaxnm.f64 d5, d22, d30
@ CHECK: vmaxnm.f64 d5, d22, d30   @ encoding: [0xae,0x5b,0x86,0xfe]
  vminnm.f32 s0, s0, s12
@ CHECK: vminnm.f32 s0, s0, s12    @ encoding: [0x46,0x0a,0x80,0xfe]
  vminnm.f64 d4, d6, d9
@ CHECK: vminnm.f64 d4, d6, d9     @ encoding: [0x49,0x4b,0x86,0xfe]

@ VRINT{Z,R,X}

  vrintzge.f64 d3, d12
@ CHECK: vrintzge.f64 d3, d12   @ encoding: [0xcc,0x3b,0xb6,0xae]
  vrintz.f32 s3, s24
@ CHECK: vrintz.f32 s3, s24     @ encoding: [0xcc,0x1a,0xf6,0xee]
  vrintrlt.f64 d5, d0
@ CHECK: vrintrlt.f64 d5, d0    @ encoding: [0x40,0x5b,0xb6,0xbe]
  vrintr.f32 s0, s9
@ CHECK: vrintr.f32 s0, s9      @ encoding: [0x64,0x0a,0xb6,0xee]
  vrintxeq.f64 d28, d30
@ CHECK: vrintxeq.f64 d28, d30  @ encoding: [0x6e,0xcb,0xf7,0x0e]
  vrintxvs.f32 s10, s14
@ CHECK: vrintxvs.f32 s10, s14  @ encoding: [0x47,0x5a,0xb7,0x6e]

@ VRINT{A,N,P,M}

  vrinta.f64 d3, d4
@ CHECK: vrinta.f64 d3, d4     @ encoding: [0x44,0x3b,0xb8,0xfe]
  vrinta.f32 s12, s1
@ CHECK: vrinta.f32 s12, s1    @ encoding: [0x60,0x6a,0xb8,0xfe]
  vrintn.f64 d3, d4
@ CHECK: vrintn.f64 d3, d4     @ encoding: [0x44,0x3b,0xb9,0xfe]
  vrintn.f32 s12, s1
@ CHECK: vrintn.f32 s12, s1    @ encoding: [0x60,0x6a,0xb9,0xfe]
  vrintp.f64 d3, d4
@ CHECK: vrintp.f64 d3, d4     @ encoding: [0x44,0x3b,0xba,0xfe]
  vrintp.f32 s12, s1
@ CHECK: vrintp.f32 s12, s1    @ encoding: [0x60,0x6a,0xba,0xfe]
  vrintm.f64 d3, d4
@ CHECK: vrintm.f64 d3, d4     @ encoding: [0x44,0x3b,0xbb,0xfe]
  vrintm.f32 s12, s1
@ CHECK: vrintm.f32 s12, s1    @ encoding: [0x60,0x6a,0xbb,0xfe]

@ MVFR2

  vmrs sp, mvfr2
@ CHECK: vmrs sp, mvfr2        @ encoding: [0x10,0xda,0xf5,0xee]
