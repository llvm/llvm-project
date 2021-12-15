@ RUN: llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16 -show-encoding < %s | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16,+thumb-mode -show-encoding < %s | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-mc -triple thumbv8.1m.main -mattr=+mve.fp,+fullfp16 -show-encoding < %s | FileCheck %s --check-prefix=THUMB
@ RUN: llvm-mc -triple thumbv8.1m.main -mattr=+fullfp16 -show-encoding < %s | FileCheck %s --check-prefix=THUMB

         vadd.f16  s0, s1, s0
@ ARM:    vadd.f16 s0, s1, s0        @ encoding: [0x80,0x09,0x30,0xee]
@ THUMB:  vadd.f16 s0, s1, s0        @ encoding: [0x30,0xee,0x80,0x09]

         vsub.f16  s0, s1, s0
@ ARM:   vsub.f16 s0, s1, s0         @ encoding: [0xc0,0x09,0x30,0xee]
@ THUMB: vsub.f16 s0, s1, s0         @ encoding: [0x30,0xee,0xc0,0x09]

         vdiv.f16  s0, s1, s0
@ ARM:   vdiv.f16 s0, s1, s0         @ encoding: [0x80,0x09,0x80,0xee]
@ THUMB: vdiv.f16 s0, s1, s0         @ encoding: [0x80,0xee,0x80,0x09]

         vmul.f16  s0, s1, s0
@ ARM:   vmul.f16 s0, s1, s0         @ encoding: [0x80,0x09,0x20,0xee]
@ THUMB: vmul.f16 s0, s1, s0         @ encoding: [0x20,0xee,0x80,0x09]

         vnmul.f16       s0, s1, s0
@ ARM:   vnmul.f16 s0, s1, s0        @ encoding: [0xc0,0x09,0x20,0xee]
@ THUMB: vnmul.f16 s0, s1, s0        @ encoding: [0x20,0xee,0xc0,0x09]

         vmla.f16        s1, s2, s0
@ ARM:   vmla.f16 s1, s2, s0         @ encoding: [0x00,0x09,0x41,0xee]
@ THUMB: vmla.f16 s1, s2, s0         @ encoding: [0x41,0xee,0x00,0x09]

         vmls.f16        s1, s2, s0
@ ARM:   vmls.f16 s1, s2, s0         @ encoding: [0x40,0x09,0x41,0xee]
@ THUMB: vmls.f16 s1, s2, s0         @ encoding: [0x41,0xee,0x40,0x09]

         vnmla.f16       s1, s2, s0
@ ARM:   vnmla.f16 s1, s2, s0        @ encoding: [0x40,0x09,0x51,0xee]
@ THUMB: vnmla.f16 s1, s2, s0        @ encoding: [0x51,0xee,0x40,0x09]

         vnmls.f16       s1, s2, s0
@ ARM:   vnmls.f16 s1, s2, s0        @ encoding: [0x00,0x09,0x51,0xee]
@ THUMB: vnmls.f16 s1, s2, s0        @ encoding: [0x51,0xee,0x00,0x09]

         vcmp.f16 s0, s1
@ ARM:   vcmp.f16        s0, s1      @ encoding: [0x60,0x09,0xb4,0xee]
@ THUMB: vcmp.f16        s0, s1      @ encoding: [0xb4,0xee,0x60,0x09]

         vcmp.f16 s2, #0
@ ARM:   vcmp.f16        s2, #0      @ encoding: [0x40,0x19,0xb5,0xee]
@ THUMB: vcmp.f16        s2, #0      @ encoding: [0xb5,0xee,0x40,0x19]

         vcmpe.f16       s1, s0
@ ARM:   vcmpe.f16 s1, s0            @ encoding: [0xc0,0x09,0xf4,0xee]
@ THUMB: vcmpe.f16 s1, s0            @ encoding: [0xf4,0xee,0xc0,0x09]

         vcmpe.f16       s0, #0
@ ARM:   vcmpe.f16 s0, #0            @ encoding: [0xc0,0x09,0xb5,0xee]
@ THUMB: vcmpe.f16 s0, #0            @ encoding: [0xb5,0xee,0xc0,0x09]

         vabs.f16        s0, s0
@ ARM:   vabs.f16 s0, s0             @ encoding: [0xc0,0x09,0xb0,0xee]
@ THUMB: vabs.f16 s0, s0             @ encoding: [0xb0,0xee,0xc0,0x09]

         vneg.f16        s0, s0
@ ARM:   vneg.f16 s0, s0             @ encoding: [0x40,0x09,0xb1,0xee]
@ THUMB: vneg.f16 s0, s0             @ encoding: [0xb1,0xee,0x40,0x09]

         vsqrt.f16       s0, s0
@ ARM:   vsqrt.f16 s0, s0            @ encoding: [0xc0,0x09,0xb1,0xee]
@ THUMB: vsqrt.f16 s0, s0            @ encoding: [0xb1,0xee,0xc0,0x09]

         vcvt.f16.s32    s0, s0
         vcvt.f16.u32    s0, s0
         vcvt.s32.f16    s0, s0
         vcvt.u32.f16    s0, s0
@ ARM:   vcvt.f16.s32 s0, s0         @ encoding: [0xc0,0x09,0xb8,0xee]
@ ARM:   vcvt.f16.u32 s0, s0         @ encoding: [0x40,0x09,0xb8,0xee]
@ ARM:   vcvt.s32.f16 s0, s0         @ encoding: [0xc0,0x09,0xbd,0xee]
@ ARM:   vcvt.u32.f16 s0, s0         @ encoding: [0xc0,0x09,0xbc,0xee]
@ THUMB: vcvt.f16.s32 s0, s0         @ encoding: [0xb8,0xee,0xc0,0x09]
@ THUMB: vcvt.f16.u32 s0, s0         @ encoding: [0xb8,0xee,0x40,0x09]
@ THUMB: vcvt.s32.f16 s0, s0         @ encoding: [0xbd,0xee,0xc0,0x09]
@ THUMB: vcvt.u32.f16 s0, s0         @ encoding: [0xbc,0xee,0xc0,0x09]

         vcvtr.s32.f16  s0, s1
         vcvtr.u32.f16  s0, s1
@ ARM:   vcvtr.s32.f16  s0, s1       @ encoding: [0x60,0x09,0xbd,0xee]
@ ARM:   vcvtr.u32.f16  s0, s1       @ encoding: [0x60,0x09,0xbc,0xee]
@ THUMB: vcvtr.s32.f16  s0, s1       @ encoding: [0xbd,0xee,0x60,0x09]
@ THUMB: vcvtr.u32.f16  s0, s1       @ encoding: [0xbc,0xee,0x60,0x09]

         vcvt.f16.u32 s0, s0, #20
         vcvt.f16.u16 s0, s0, #1
         vcvt.f16.s32 s1, s1, #20
         vcvt.f16.s16 s17, s17, #1
         vcvt.u32.f16 s12, s12, #20
         vcvt.u16.f16 s28, s28, #1
         vcvt.s32.f16 s1, s1, #20
         vcvt.s16.f16 s17, s17, #1
@ ARM:   vcvt.f16.u32   s0, s0, #20     @ encoding: [0xc6,0x09,0xbb,0xee]
@ ARM:   vcvt.f16.u16   s0, s0, #1      @ encoding: [0x67,0x09,0xbb,0xee]
@ ARM:   vcvt.f16.s32   s1, s1, #20     @ encoding: [0xc6,0x09,0xfa,0xee]
@ ARM:   vcvt.f16.s16   s17, s17, #1    @ encoding: [0x67,0x89,0xfa,0xee]
@ ARM:   vcvt.u32.f16   s12, s12, #20   @ encoding: [0xc6,0x69,0xbf,0xee]
@ ARM:   vcvt.u16.f16   s28, s28, #1    @ encoding: [0x67,0xe9,0xbf,0xee]
@ ARM:   vcvt.s32.f16   s1, s1, #20     @ encoding: [0xc6,0x09,0xfe,0xee]
@ ARM:   vcvt.s16.f16   s17, s17, #1    @ encoding: [0x67,0x89,0xfe,0xee]
@ THUMB: vcvt.f16.u32   s0, s0, #20     @ encoding: [0xbb,0xee,0xc6,0x09]
@ THUMB: vcvt.f16.u16   s0, s0, #1      @ encoding: [0xbb,0xee,0x67,0x09]
@ THUMB: vcvt.f16.s32   s1, s1, #20     @ encoding: [0xfa,0xee,0xc6,0x09]
@ THUMB: vcvt.f16.s16   s17, s17, #1    @ encoding: [0xfa,0xee,0x67,0x89]
@ THUMB: vcvt.u32.f16   s12, s12, #20   @ encoding: [0xbf,0xee,0xc6,0x69]
@ THUMB: vcvt.u16.f16   s28, s28, #1    @ encoding: [0xbf,0xee,0x67,0xe9]
@ THUMB: vcvt.s32.f16   s1, s1, #20     @ encoding: [0xfe,0xee,0xc6,0x09]
@ THUMB: vcvt.s16.f16   s17, s17, #1    @ encoding: [0xfe,0xee,0x67,0x89]

  vcvta.s32.f16 s2, s3
@ ARM:   vcvta.s32.f16 s2, s3     @ encoding: [0xe1,0x19,0xbc,0xfe]
@ THUMB: vcvta.s32.f16 s2, s3     @ encoding: [0xbc,0xfe,0xe1,0x19]

  vcvtn.s32.f16 s6, s23
@ ARM:   vcvtn.s32.f16 s6, s23     @ encoding: [0xeb,0x39,0xbd,0xfe]
@ THUMB: vcvtn.s32.f16 s6, s23     @ encoding: [0xbd,0xfe,0xeb,0x39]

  vcvtp.s32.f16 s0, s4
@ ARM:   vcvtp.s32.f16 s0, s4     @ encoding: [0xc2,0x09,0xbe,0xfe]
@ THUMB: vcvtp.s32.f16 s0, s4     @ encoding: [0xbe,0xfe,0xc2,0x09]

  vcvtm.s32.f16 s17, s8
@ ARM:   vcvtm.s32.f16 s17, s8     @ encoding: [0xc4,0x89,0xff,0xfe]
@ THUMB: vcvtm.s32.f16 s17, s8     @ encoding: [0xff,0xfe,0xc4,0x89]

  vcvta.u32.f16 s2, s3
@ ARM:   vcvta.u32.f16 s2, s3     @ encoding: [0x61,0x19,0xbc,0xfe]
@ THUMB: vcvta.u32.f16 s2, s3     @ encoding: [0xbc,0xfe,0x61,0x19]

  vcvtn.u32.f16 s6, s23
@ ARM:   vcvtn.u32.f16 s6, s23     @ encoding: [0x6b,0x39,0xbd,0xfe]
@ THUMB: vcvtn.u32.f16 s6, s23     @ encoding: [0xbd,0xfe,0x6b,0x39]

  vcvtp.u32.f16 s0, s4
@ ARM:   vcvtp.u32.f16 s0, s4     @ encoding: [0x42,0x09,0xbe,0xfe]
@ THUMB: vcvtp.u32.f16 s0, s4     @ encoding: [0xbe,0xfe,0x42,0x09]

  vcvtm.u32.f16 s17, s8
@ ARM:   vcvtm.u32.f16 s17, s8     @ encoding: [0x44,0x89,0xff,0xfe]
@ THUMB: vcvtm.u32.f16 s17, s8     @ encoding: [0xff,0xfe,0x44,0x89]

  vselge.f16 s4, s1, s23
@ ARM:   vselge.f16 s4, s1, s23    @ encoding: [0xab,0x29,0x20,0xfe]
@ THUMB: vselge.f16 s4, s1, s23    @ encoding: [0x20,0xfe,0xab,0x29]

  vselgt.f16 s0, s1, s0
@ ARM:   vselgt.f16 s0, s1, s0    @ encoding: [0x80,0x09,0x30,0xfe]
@ THUMB: vselgt.f16 s0, s1, s0    @ encoding: [0x30,0xfe,0x80,0x09]

  vseleq.f16 s30, s28, s23
@ ARM:   vseleq.f16 s30, s28, s23 @ encoding: [0x2b,0xf9,0x0e,0xfe]
@ THUMB: vseleq.f16 s30, s28, s23 @ encoding: [0x0e,0xfe,0x2b,0xf9]

  vselvs.f16 s21, s16, s14
@ ARM:   vselvs.f16 s21, s16, s14 @ encoding: [0x07,0xa9,0x58,0xfe]
@ THUMB: vselvs.f16 s21, s16, s14 @ encoding: [0x58,0xfe,0x07,0xa9]

  vmaxnm.f16 s5, s12, s0
@ ARM:   vmaxnm.f16 s5, s12, s0    @ encoding: [0x00,0x29,0xc6,0xfe]
@ THUMB: vmaxnm.f16 s5, s12, s0    @ encoding: [0xc6,0xfe,0x00,0x29]

  vminnm.f16 s0, s0, s12
@ ARM:   vminnm.f16 s0, s0, s12    @ encoding: [0x46,0x09,0x80,0xfe]
@ THUMB: vminnm.f16 s0, s0, s12    @ encoding: [0x80,0xfe,0x46,0x09]

  vrintz.f16 s3, s24
  vrintz.f16.f16 s3, s24
@ ARM:   vrintz.f16 s3, s24     @ encoding: [0xcc,0x19,0xf6,0xee]
@ ARM:   vrintz.f16 s3, s24     @ encoding: [0xcc,0x19,0xf6,0xee]
@ THUMB: vrintz.f16 s3, s24     @ encoding: [0xf6,0xee,0xcc,0x19]
@ THUMB: vrintz.f16 s3, s24     @ encoding: [0xf6,0xee,0xcc,0x19]

  vrintr.f16 s0, s9
  vrintr.f16.f16 s0, s9
@ ARM:   vrintr.f16 s0, s9      @ encoding: [0x64,0x09,0xb6,0xee]
@ ARM:   vrintr.f16 s0, s9      @ encoding: [0x64,0x09,0xb6,0xee]
@ THUMB: vrintr.f16 s0, s9      @ encoding: [0xb6,0xee,0x64,0x09]
@ THUMB: vrintr.f16 s0, s9      @ encoding: [0xb6,0xee,0x64,0x09]

  vrintx.f16 s10, s14
  vrintx.f16.f16 s10, s14
@ ARM:   vrintx.f16 s10, s14  @ encoding: [0x47,0x59,0xb7,0xee]
@ ARM:   vrintx.f16 s10, s14  @ encoding: [0x47,0x59,0xb7,0xee]
@ THUMB: vrintx.f16 s10, s14  @ encoding: [0xb7,0xee,0x47,0x59]
@ THUMB: vrintx.f16 s10, s14  @ encoding: [0xb7,0xee,0x47,0x59]

  vrinta.f16 s12, s1
  vrinta.f16.f16 s12, s1
@ ARM:   vrinta.f16 s12, s1    @ encoding: [0x60,0x69,0xb8,0xfe]
@ ARM:   vrinta.f16 s12, s1    @ encoding: [0x60,0x69,0xb8,0xfe]
@ THUMB: vrinta.f16 s12, s1    @ encoding: [0xb8,0xfe,0x60,0x69]
@ THUMB: vrinta.f16 s12, s1    @ encoding: [0xb8,0xfe,0x60,0x69]

  vrintn.f16 s12, s1
  vrintn.f16.f16 s12, s1
@ ARM:   vrintn.f16 s12, s1    @ encoding: [0x60,0x69,0xb9,0xfe]
@ ARM:   vrintn.f16 s12, s1    @ encoding: [0x60,0x69,0xb9,0xfe]
@ THUMB: vrintn.f16 s12, s1    @ encoding: [0xb9,0xfe,0x60,0x69]
@ THUMB: vrintn.f16 s12, s1    @ encoding: [0xb9,0xfe,0x60,0x69]

  vrintp.f16 s12, s1
  vrintp.f16.f16 s12, s1
@ ARM:   vrintp.f16 s12, s1    @ encoding: [0x60,0x69,0xba,0xfe]
@ ARM:   vrintp.f16 s12, s1    @ encoding: [0x60,0x69,0xba,0xfe]
@ THUMB: vrintp.f16 s12, s1    @ encoding: [0xba,0xfe,0x60,0x69]
@ THUMB: vrintp.f16 s12, s1    @ encoding: [0xba,0xfe,0x60,0x69]

  vrintm.f16 s12, s1
  vrintm.f16.f16 s12, s1
@ ARM:   vrintm.f16 s12, s1    @ encoding: [0x60,0x69,0xbb,0xfe]
@ ARM:   vrintm.f16 s12, s1    @ encoding: [0x60,0x69,0xbb,0xfe]
@ THUMB: vrintm.f16 s12, s1    @ encoding: [0xbb,0xfe,0x60,0x69]
@ THUMB: vrintm.f16 s12, s1    @ encoding: [0xbb,0xfe,0x60,0x69]

  vfma.f16 s2, s7, s4
@ ARM:   vfma.f16        s2, s7, s4      @ encoding: [0x82,0x19,0xa3,0xee]
@ THUMB: vfma.f16        s2, s7, s4      @ encoding: [0xa3,0xee,0x82,0x19]

  vfms.f16 s2, s7, s4
@ ARM:   vfms.f16        s2, s7, s4      @ encoding: [0xc2,0x19,0xa3,0xee]
@ THUMB: vfms.f16        s2, s7, s4      @ encoding: [0xa3,0xee,0xc2,0x19]

  vfnma.f16 s2, s7, s4
@ ARM:   vfnma.f16       s2, s7, s4      @ encoding: [0xc2,0x19,0x93,0xee]
@ THUMB: vfnma.f16       s2, s7, s4      @ encoding: [0x93,0xee,0xc2,0x19]

  vfnms.f16 s2, s7, s4
@ ARM:   vfnms.f16       s2, s7, s4      @ encoding: [0x82,0x19,0x93,0xee]
@ THUMB: vfnms.f16       s2, s7, s4      @ encoding: [0x93,0xee,0x82,0x19]

  vmovx.f16 s2, s5
  vins.f16 s2, s5
@ ARM:   vmovx.f16       s2, s5          @ encoding: [0x62,0x1a,0xb0,0xfe]
@ ARM:   vins.f16        s2, s5          @ encoding: [0xe2,0x1a,0xb0,0xfe]
@ THUMB: vmovx.f16       s2, s5          @ encoding: [0xb0,0xfe,0x62,0x1a]
@ THUMB: vins.f16        s2, s5          @ encoding: [0xb0,0xfe,0xe2,0x1a]


  vldr.16 s1, [pc, #6]
  vldr.16 s2, [pc, #510]
  vldr.16 s3, [pc, #-510]
  vldr.16 s4, [r4, #-18]
@ ARM:   vldr.16 s1, [pc, #6]          @ encoding: [0x03,0x09,0xdf,0xed]
@ ARM:   vldr.16 s2, [pc, #510]        @ encoding: [0xff,0x19,0x9f,0xed]
@ ARM:   vldr.16 s3, [pc, #-510]       @ encoding: [0xff,0x19,0x5f,0xed]
@ ARM:   vldr.16 s4, [r4, #-18]        @ encoding: [0x09,0x29,0x14,0xed]
@ THUMB: vldr.16 s1, [pc, #6]          @ encoding: [0xdf,0xed,0x03,0x09]
@ THUMB: vldr.16 s2, [pc, #510]        @ encoding: [0x9f,0xed,0xff,0x19]
@ THUMB: vldr.16 s3, [pc, #-510]       @ encoding: [0x5f,0xed,0xff,0x19]
@ THUMB: vldr.16 s4, [r4, #-18]        @ encoding: [0x14,0xed,0x09,0x29]


  vstr.16 s1, [pc, #6]
  vstr.16 s2, [pc, #510]
  vstr.16 s3, [pc, #-510]
  vstr.16 s4, [r4, #-18]
@ ARM:   vstr.16 s1, [pc, #6]          @ encoding: [0x03,0x09,0xcf,0xed]
@ ARM:   vstr.16 s2, [pc, #510]        @ encoding: [0xff,0x19,0x8f,0xed]
@ ARM:   vstr.16 s3, [pc, #-510]       @ encoding: [0xff,0x19,0x4f,0xed]
@ ARM:   vstr.16 s4, [r4, #-18]        @ encoding: [0x09,0x29,0x04,0xed]
@ THUMB: vstr.16 s1, [pc, #6]          @ encoding: [0xcf,0xed,0x03,0x09]
@ THUMB: vstr.16 s2, [pc, #510]        @ encoding: [0x8f,0xed,0xff,0x19]
@ THUMB: vstr.16 s3, [pc, #-510]       @ encoding: [0x4f,0xed,0xff,0x19]
@ THUMB: vstr.16 s4, [r4, #-18]        @ encoding: [0x04,0xed,0x09,0x29]

  vmov.f16 s0, #1.0
@ ARM:   vmov.f16        s0, #1.000000e+00 @ encoding: [0x00,0x09,0xb7,0xee]
@ THUMB: vmov.f16        s0, #1.000000e+00 @ encoding: [0xb7,0xee,0x00,0x09]

  vmov.f16 s1, r2
  vmov.f16 r3, s4
@ ARM:   vmov.f16        s1, r2          @ encoding: [0x90,0x29,0x00,0xee]
@ ARM:   vmov.f16        r3, s4          @ encoding: [0x10,0x39,0x12,0xee]
@ THUMB: vmov.f16       s1, r2          @ encoding: [0x00,0xee,0x90,0x29]
@ THUMB: vmov.f16       r3, s4          @ encoding: [0x12,0xee,0x10,0x39]
