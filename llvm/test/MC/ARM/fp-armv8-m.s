@ RUN: llvm-mc -triple thumbv8.1m.main -mattr=+mve.fp -mattr=+fp64 -show-encoding < %s \
@ RUN:     | FileCheck --check-prefix=CHECK-V81M %s
@ RUN: llvm-mc -triple thumbv8.1m.main -mattr=+fp-armv8 -mattr=+fp64 -show-encoding < %s \
@ RUN:     | FileCheck --check-prefix=CHECK-V81M %s

@ VCVT{B,T}

  vcvtt.f64.f16 d3, s1
@ CHECK-V81M: vcvtt.f64.f16 d3, s1      @ encoding: [0xb2,0xee,0xe0,0x3b]
  vcvtt.f16.f64 s5, d12
@ CHECK-V81M: vcvtt.f16.f64 s5, d12     @ encoding: [0xf3,0xee,0xcc,0x2b]

  vcvtb.f64.f16 d3, s1
@ CHECK-V81M: vcvtb.f64.f16 d3, s1     @ encoding: [0xb2,0xee,0x60,0x3b]
  vcvtb.f16.f64 s4, d1
@ CHECK-V81M: vcvtb.f16.f64 s4, d1     @ encoding: [0xb3,0xee,0x41,0x2b]

  it ge
  vcvttge.f64.f16 d3, s1
@ CHECK-V81M: vcvttge.f64.f16 d3, s1      @ encoding: [0xb2,0xee,0xe0,0x3b]
  it gt
  vcvttgt.f16.f64 s5, d12
@ CHECK-V81M: vcvttgt.f16.f64 s5, d12     @ encoding: [0xf3,0xee,0xcc,0x2b]

  it eq
  vcvtbeq.f64.f16 d3, s1
@ CHECK-V81M: vcvtbeq.f64.f16 d3, s1     @ encoding: [0xb2,0xee,0x60,0x3b]
  it lt
  vcvtblt.f16.f64 s4, d1
@ CHECK-V81M: vcvtblt.f16.f64 s4, d1     @ encoding: [0xb3,0xee,0x41,0x2b]


@ VCVT{A,N,P,M}

  vcvta.s32.f32 s2, s3
@ CHECK-V81M: vcvta.s32.f32 s2, s3     @ encoding: [0xbc,0xfe,0xe1,0x1a]
  vcvta.s32.f64 s2, d3
@ CHECK-V81M: vcvta.s32.f64 s2, d3     @ encoding: [0xbc,0xfe,0xc3,0x1b]
  vcvtn.s32.f32 s6, s23
@ CHECK-V81M: vcvtn.s32.f32 s6, s23     @ encoding: [0xbd,0xfe,0xeb,0x3a]
  vcvtn.s32.f64 s6, d7
@ CHECK-V81M: vcvtn.s32.f64 s6, d7     @ encoding: [0xbd,0xfe,0xc7,0x3b]
  vcvtp.s32.f32 s0, s4
@ CHECK-V81M: vcvtp.s32.f32 s0, s4     @ encoding: [0xbe,0xfe,0xc2,0x0a]
  vcvtp.s32.f64 s0, d4
@ CHECK-V81M: vcvtp.s32.f64 s0, d4     @ encoding: [0xbe,0xfe,0xc4,0x0b]
  vcvtm.s32.f32 s17, s8
@ CHECK-V81M: vcvtm.s32.f32 s17, s8     @ encoding: [0xff,0xfe,0xc4,0x8a]
  vcvtm.s32.f64 s17, d8
@ CHECK-V81M: vcvtm.s32.f64 s17, d8     @ encoding: [0xff,0xfe,0xc8,0x8b]

  vcvta.u32.f32 s2, s3
@ CHECK-V81M: vcvta.u32.f32 s2, s3     @ encoding: [0xbc,0xfe,0x61,0x1a]
  vcvta.u32.f64 s2, d3
@ CHECK-V81M: vcvta.u32.f64 s2, d3     @ encoding: [0xbc,0xfe,0x43,0x1b]
  vcvtn.u32.f32 s6, s23
@ CHECK-V81M: vcvtn.u32.f32 s6, s23     @ encoding: [0xbd,0xfe,0x6b,0x3a]
  vcvtn.u32.f64 s6, d7
@ CHECK-V81M: vcvtn.u32.f64 s6, d7     @ encoding: [0xbd,0xfe,0x47,0x3b]
  vcvtp.u32.f32 s0, s4
@ CHECK-V81M: vcvtp.u32.f32 s0, s4     @ encoding: [0xbe,0xfe,0x42,0x0a]
  vcvtp.u32.f64 s0, d4
@ CHECK-V81M: vcvtp.u32.f64 s0, d4     @ encoding: [0xbe,0xfe,0x44,0x0b]
  vcvtm.u32.f32 s17, s8
@ CHECK-V81M: vcvtm.u32.f32 s17, s8     @ encoding: [0xff,0xfe,0x44,0x8a]
  vcvtm.u32.f64 s17, d8
@ CHECK-V81M: vcvtm.u32.f64 s17, d8     @ encoding: [0xff,0xfe,0x48,0x8b]


@ VSEL
  vselge.f32 s4, s1, s23
@ CHECK-V81M: vselge.f32 s4, s1, s23    @ encoding: [0x20,0xfe,0xab,0x2a]
  vselge.f64 d0, d1, d3
@ CHECK-V81M: vselge.f64 d0, d1, d3  @ encoding: [0x21,0xfe,0x03,0x0b]
  vselgt.f32 s0, s1, s0
@ CHECK-V81M: vselgt.f32 s0, s1, s0    @ encoding: [0x30,0xfe,0x80,0x0a]
  vselgt.f64 d5, d10, d11
@ CHECK-V81M: vselgt.f64 d5, d10, d11  @ encoding: [0x3a,0xfe,0x0b,0x5b]
  vseleq.f32 s30, s28, s23
@ CHECK-V81M: vseleq.f32 s30, s28, s23 @ encoding: [0x0e,0xfe,0x2b,0xfa]
  vseleq.f64 d2, d4, d8
@ CHECK-V81M: vseleq.f64 d2, d4, d8    @ encoding: [0x04,0xfe,0x08,0x2b]
  vselvs.f32 s21, s16, s14
@ CHECK-V81M: vselvs.f32 s21, s16, s14 @ encoding: [0x58,0xfe,0x07,0xaa]
  vselvs.f64 d0, d1, d15
@ CHECK-V81M: vselvs.f64 d0, d1, d15   @ encoding: [0x11,0xfe,0x0f,0x0b]


@ VMAXNM / VMINNM
  vmaxnm.f32 s5, s12, s0
@ CHECK-V81M: vmaxnm.f32 s5, s12, s0    @ encoding: [0xc6,0xfe,0x00,0x2a]
  vmaxnm.f64 d5, d14, d15
@ CHECK-V81M: vmaxnm.f64 d5, d14, d15   @ encoding: [0x8e,0xfe,0x0f,0x5b]
  vminnm.f32 s0, s0, s12
@ CHECK-V81M: vminnm.f32 s0, s0, s12    @ encoding: [0x80,0xfe,0x46,0x0a]
  vminnm.f64 d4, d6, d9
@ CHECK-V81M: vminnm.f64 d4, d6, d9     @ encoding: [0x86,0xfe,0x49,0x4b]

@ VRINT{Z,R,X}

  it ge
  vrintzge.f64 d3, d12
@ CHECK-V81M: vrintzge.f64 d3, d12   @ encoding: [0xb6,0xee,0xcc,0x3b]
  vrintz.f32 s3, s24
@ CHECK-V81M: vrintz.f32 s3, s24     @ encoding: [0xf6,0xee,0xcc,0x1a]
  it lt
  vrintrlt.f64 d5, d0
@ CHECK-V81M: vrintrlt.f64 d5, d0    @ encoding: [0xb6,0xee,0x40,0x5b]
  vrintr.f32 s0, s9
@ CHECK-V81M: vrintr.f32 s0, s9      @ encoding: [0xb6,0xee,0x64,0x0a]
  it eq
  vrintxeq.f64 d14, d15
@ CHECK-V81M: vrintxeq.f64 d14, d15  @ encoding: [0xb7,0xee,0x4f,0xeb]
  it vs
  vrintxvs.f32 s10, s14
@ CHECK-V81M: vrintxvs.f32 s10, s14  @ encoding: [0xb7,0xee,0x47,0x5a]

@ VRINT{A,N,P,M}

  vrinta.f64 d3, d4
@ CHECK-V81M: vrinta.f64 d3, d4     @ encoding: [0xb8,0xfe,0x44,0x3b]
  vrinta.f32 s12, s1
@ CHECK-V81M: vrinta.f32 s12, s1    @ encoding: [0xb8,0xfe,0x60,0x6a]
  vrintn.f64 d3, d4
@ CHECK-V81M: vrintn.f64 d3, d4     @ encoding: [0xb9,0xfe,0x44,0x3b]
  vrintn.f32 s12, s1
@ CHECK-V81M: vrintn.f32 s12, s1    @ encoding: [0xb9,0xfe,0x60,0x6a]
  vrintp.f64 d3, d4
@ CHECK-V81M: vrintp.f64 d3, d4     @ encoding: [0xba,0xfe,0x44,0x3b]
  vrintp.f32 s12, s1
@ CHECK-V81M: vrintp.f32 s12, s1    @ encoding: [0xba,0xfe,0x60,0x6a]
  vrintm.f64 d3, d4
@ CHECK-V81M: vrintm.f64 d3, d4     @ encoding: [0xbb,0xfe,0x44,0x3b]
  vrintm.f32 s12, s1
@ CHECK-V81M: vrintm.f32 s12, s1    @ encoding: [0xbb,0xfe,0x60,0x6a]

@ MVFR2

  vmrs sp, mvfr2
@ CHECK-V81M: vmrs sp, mvfr2        @ encoding: [0xf5,0xee,0x10,0xda]
