@ RUN: llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16,+neon -show-encoding < %s | FileCheck %s --check-prefix=ARM
@ RUN: llvm-mc -triple thumbv8a-none-eabi -mattr=+fullfp16,+neon -show-encoding < %s | FileCheck %s --check-prefix=THUMB

  vadd.f16 d0, d1, d2
  vadd.f16 q0, q1, q2
@ ARM:   vadd.f16        d0, d1, d2      @ encoding: [0x02,0x0d,0x11,0xf2]
@ ARM:   vadd.f16        q0, q1, q2      @ encoding: [0x44,0x0d,0x12,0xf2]
@ THUMB: vadd.f16        d0, d1, d2      @ encoding: [0x11,0xef,0x02,0x0d]
@ THUMB: vadd.f16        q0, q1, q2      @ encoding: [0x12,0xef,0x44,0x0d]

  vsub.f16 d0, d1, d2
  vsub.f16 q0, q1, q2
@ ARM:   vsub.f16        d0, d1, d2      @ encoding: [0x02,0x0d,0x31,0xf2]
@ ARM:   vsub.f16        q0, q1, q2      @ encoding: [0x44,0x0d,0x32,0xf2]
@ THUMB: vsub.f16        d0, d1, d2      @ encoding: [0x31,0xef,0x02,0x0d]
@ THUMB: vsub.f16        q0, q1, q2      @ encoding: [0x32,0xef,0x44,0x0d]

  vmul.f16 d0, d1, d2
  vmul.f16 q0, q1, q2
@ ARM:   vmul.f16        d0, d1, d2      @ encoding: [0x12,0x0d,0x11,0xf3]
@ ARM:   vmul.f16        q0, q1, q2      @ encoding: [0x54,0x0d,0x12,0xf3]
@ THUMB: vmul.f16        d0, d1, d2      @ encoding: [0x11,0xff,0x12,0x0d]
@ THUMB: vmul.f16        q0, q1, q2      @ encoding: [0x12,0xff,0x54,0x0d]

  vmul.f16 d1, d2, d3[2]
  vmul.f16 q4, q5, d6[3]
@ ARM:   vmul.f16        d1, d2, d3[2]   @ encoding: [0x63,0x19,0x92,0xf2]
@ ARM:   vmul.f16        q4, q5, d6[3]   @ encoding: [0x6e,0x89,0x9a,0xf3]
@ THUMB: vmul.f16        d1, d2, d3[2]   @ encoding: [0x92,0xef,0x63,0x19]
@ THUMB: vmul.f16        q4, q5, d6[3]   @ encoding: [0x9a,0xff,0x6e,0x89]

  vmla.f16 d0, d1, d2
  vmla.f16 q0, q1, q2
@ ARM:   vmla.f16        d0, d1, d2      @ encoding: [0x12,0x0d,0x11,0xf2]
@ ARM:   vmla.f16        q0, q1, q2      @ encoding: [0x54,0x0d,0x12,0xf2]
@ THUMB: vmla.f16        d0, d1, d2      @ encoding: [0x11,0xef,0x12,0x0d]
@ THUMB: vmla.f16        q0, q1, q2      @ encoding: [0x12,0xef,0x54,0x0d]

  vmla.f16 d5, d6, d7[2]
  vmla.f16 q5, q6, d7[3]
@ ARM:   vmla.f16        d5, d6, d7[2]   @ encoding: [0x67,0x51,0x96,0xf2]
@ ARM:   vmla.f16        q5, q6, d7[3]   @ encoding: [0x6f,0xa1,0x9c,0xf3]
@ THUMB: vmla.f16        d5, d6, d7[2]   @ encoding: [0x96,0xef,0x67,0x51]
@ THUMB: vmla.f16        q5, q6, d7[3]   @ encoding: [0x9c,0xff,0x6f,0xa1]

  vmls.f16 d0, d1, d2
  vmls.f16 q0, q1, q2
@ ARM:   vmls.f16        d0, d1, d2      @ encoding: [0x12,0x0d,0x31,0xf2]
@ ARM:   vmls.f16        q0, q1, q2      @ encoding: [0x54,0x0d,0x32,0xf2]
@ THUMB: vmls.f16        d0, d1, d2      @ encoding: [0x31,0xef,0x12,0x0d]
@ THUMB: vmls.f16        q0, q1, q2      @ encoding: [0x32,0xef,0x54,0x0d]

  vmls.f16 d5, d6, d7[2]
  vmls.f16 q5, q6, d7[3]
@ ARM:   vmls.f16        d5, d6, d7[2]   @ encoding: [0x67,0x55,0x96,0xf2]
@ ARM:   vmls.f16        q5, q6, d7[3]   @ encoding: [0x6f,0xa5,0x9c,0xf3]
@ THUMB: vmls.f16        d5, d6, d7[2]   @ encoding: [0x96,0xef,0x67,0x55]
@ THUMB: vmls.f16        q5, q6, d7[3]   @ encoding: [0x9c,0xff,0x6f,0xa5]

  vfma.f16 d0, d1, d2
  vfma.f16 q0, q1, q2
@ ARM:   vfma.f16        d0, d1, d2      @ encoding: [0x12,0x0c,0x11,0xf2]
@ ARM:   vfma.f16        q0, q1, q2      @ encoding: [0x54,0x0c,0x12,0xf2]
@ THUMB: vfma.f16        d0, d1, d2      @ encoding: [0x11,0xef,0x12,0x0c]
@ THUMB: vfma.f16        q0, q1, q2      @ encoding: [0x12,0xef,0x54,0x0c]

  vfms.f16 d0, d1, d2
  vfms.f16 q0, q1, q2
@ ARM:   vfms.f16        d0, d1, d2      @ encoding: [0x12,0x0c,0x31,0xf2]
@ ARM:   vfms.f16        q0, q1, q2      @ encoding: [0x54,0x0c,0x32,0xf2]
@ THUMB: vfms.f16        d0, d1, d2      @ encoding: [0x31,0xef,0x12,0x0c]
@ THUMB: vfms.f16        q0, q1, q2      @ encoding: [0x32,0xef,0x54,0x0c]

  vceq.f16 d2, d3, d4
  vceq.f16 q2, q3, q4
@ ARM:   vceq.f16        d2, d3, d4      @ encoding: [0x04,0x2e,0x13,0xf2]
@ ARM:   vceq.f16        q2, q3, q4      @ encoding: [0x48,0x4e,0x16,0xf2]
@ THUMB: vceq.f16        d2, d3, d4      @ encoding: [0x13,0xef,0x04,0x2e]
@ THUMB: vceq.f16        q2, q3, q4      @ encoding: [0x16,0xef,0x48,0x4e]

  vceq.f16 d2, d3, #0
  vceq.f16 q2, q3, #0
@ ARM:   vceq.f16        d2, d3, #0      @ encoding: [0x03,0x25,0xb5,0xf3]
@ ARM:   vceq.f16        q2, q3, #0      @ encoding: [0x46,0x45,0xb5,0xf3]
@ THUMB: vceq.f16        d2, d3, #0      @ encoding: [0xb5,0xff,0x03,0x25]
@ THUMB: vceq.f16        q2, q3, #0      @ encoding: [0xb5,0xff,0x46,0x45]

  vcge.f16 d2, d3, d4
  vcge.f16 q2, q3, q4
@ ARM:   vcge.f16        d2, d3, d4      @ encoding: [0x04,0x2e,0x13,0xf3]
@ ARM:   vcge.f16        q2, q3, q4      @ encoding: [0x48,0x4e,0x16,0xf3]
@ THUMB: vcge.f16        d2, d3, d4      @ encoding: [0x13,0xff,0x04,0x2e]
@ THUMB: vcge.f16        q2, q3, q4      @ encoding: [0x16,0xff,0x48,0x4e]

  vcge.f16 d2, d3, #0
  vcge.f16 q2, q3, #0
@ ARM:   vcge.f16        d2, d3, #0      @ encoding: [0x83,0x24,0xb5,0xf3]
@ ARM:   vcge.f16        q2, q3, #0      @ encoding: [0xc6,0x44,0xb5,0xf3]
@ THUMB: vcge.f16        d2, d3, #0      @ encoding: [0xb5,0xff,0x83,0x24]
@ THUMB: vcge.f16        q2, q3, #0      @ encoding: [0xb5,0xff,0xc6,0x44]

  vcgt.f16 d2, d3, d4
  vcgt.f16 q2, q3, q4
@ ARM:   vcgt.f16        d2, d3, d4      @ encoding: [0x04,0x2e,0x33,0xf3]
@ ARM:   vcgt.f16        q2, q3, q4      @ encoding: [0x48,0x4e,0x36,0xf3]
@ THUMB: vcgt.f16        d2, d3, d4      @ encoding: [0x33,0xff,0x04,0x2e]
@ THUMB: vcgt.f16        q2, q3, q4      @ encoding: [0x36,0xff,0x48,0x4e]

  vcgt.f16 d2, d3, #0
  vcgt.f16 q2, q3, #0
@ ARM:   vcgt.f16        d2, d3, #0      @ encoding: [0x03,0x24,0xb5,0xf3]
@ ARM:   vcgt.f16        q2, q3, #0      @ encoding: [0x46,0x44,0xb5,0xf3]
@ THUMB: vcgt.f16        d2, d3, #0      @ encoding: [0xb5,0xff,0x03,0x24]
@ THUMB: vcgt.f16        q2, q3, #0      @ encoding: [0xb5,0xff,0x46,0x44]

  vcle.f16 d2, d3, d4
  vcle.f16 q2, q3, q4
@ ARM:   vcge.f16        d2, d4, d3      @ encoding: [0x03,0x2e,0x14,0xf3]
@ ARM:   vcge.f16        q2, q4, q3      @ encoding: [0x46,0x4e,0x18,0xf3]
@ THUMB: vcge.f16        d2, d4, d3      @ encoding: [0x14,0xff,0x03,0x2e]
@ THUMB: vcge.f16        q2, q4, q3      @ encoding: [0x18,0xff,0x46,0x4e]

  vcle.f16 d2, d3, #0
  vcle.f16 q2, q3, #0
@ ARM:   vcle.f16        d2, d3, #0      @ encoding: [0x83,0x25,0xb5,0xf3]
@ ARM:   vcle.f16        q2, q3, #0      @ encoding: [0xc6,0x45,0xb5,0xf3]
@ THUMB: vcle.f16        d2, d3, #0      @ encoding: [0xb5,0xff,0x83,0x25]
@ THUMB: vcle.f16        q2, q3, #0      @ encoding: [0xb5,0xff,0xc6,0x45]

  vclt.f16 d2, d3, d4
  vclt.f16 q2, q3, q4
@ ARM:   vcgt.f16        d2, d4, d3      @ encoding: [0x03,0x2e,0x34,0xf3]
@ ARM:   vcgt.f16        q2, q4, q3      @ encoding: [0x46,0x4e,0x38,0xf3]
@ THUMB: vcgt.f16        d2, d4, d3      @ encoding: [0x34,0xff,0x03,0x2e]
@ THUMB: vcgt.f16        q2, q4, q3      @ encoding: [0x38,0xff,0x46,0x4e]

  vclt.f16 d2, d3, #0
  vclt.f16 q2, q3, #0
@ ARM:   vclt.f16        d2, d3, #0      @ encoding: [0x03,0x26,0xb5,0xf3]
@ ARM:   vclt.f16        q2, q3, #0      @ encoding: [0x46,0x46,0xb5,0xf3]
@ THUMB: vclt.f16        d2, d3, #0      @ encoding: [0xb5,0xff,0x03,0x26]
@ THUMB: vclt.f16        q2, q3, #0      @ encoding: [0xb5,0xff,0x46,0x46]

  vacge.f16 d0, d1, d2
  vacge.f16 q0, q1, q2
@ ARM:   vacge.f16       d0, d1, d2      @ encoding: [0x12,0x0e,0x11,0xf3]
@ ARM:   vacge.f16       q0, q1, q2      @ encoding: [0x54,0x0e,0x12,0xf3]
@ THUMB: vacge.f16       d0, d1, d2      @ encoding: [0x11,0xff,0x12,0x0e]
@ THUMB: vacge.f16       q0, q1, q2      @ encoding: [0x12,0xff,0x54,0x0e]

  vacgt.f16 d0, d1, d2
  vacgt.f16 q0, q1, q2
@ ARM:   vacgt.f16       d0, d1, d2      @ encoding: [0x12,0x0e,0x31,0xf3]
@ ARM:   vacgt.f16       q0, q1, q2      @ encoding: [0x54,0x0e,0x32,0xf3]
@ THUMB: vacgt.f16       d0, d1, d2      @ encoding: [0x31,0xff,0x12,0x0e]
@ THUMB: vacgt.f16       q0, q1, q2      @ encoding: [0x32,0xff,0x54,0x0e]

  vacle.f16 d0, d1, d2
  vacle.f16 q0, q1, q2
@ ARM:   vacge.f16       d0, d2, d1      @ encoding: [0x11,0x0e,0x12,0xf3]
@ ARM:   vacge.f16       q0, q2, q1      @ encoding: [0x52,0x0e,0x14,0xf3]
@ THUMB: vacge.f16       d0, d2, d1      @ encoding: [0x12,0xff,0x11,0x0e]
@ THUMB: vacge.f16       q0, q2, q1      @ encoding: [0x14,0xff,0x52,0x0e]

  vaclt.f16 d0, d1, d2
  vaclt.f16 q0, q1, q2
@ ARM:   vacgt.f16       d0, d2, d1      @ encoding: [0x11,0x0e,0x32,0xf3]
@ ARM:   vacgt.f16       q0, q2, q1      @ encoding: [0x52,0x0e,0x34,0xf3]
@ THUMB: vacgt.f16       d0, d2, d1      @ encoding: [0x32,0xff,0x11,0x0e]
@ THUMB: vacgt.f16       q0, q2, q1      @ encoding: [0x34,0xff,0x52,0x0e]

  vabd.f16 d0, d1, d2
  vabd.f16 q0, q1, q2
@ ARM:   vabd.f16        d0, d1, d2      @ encoding: [0x02,0x0d,0x31,0xf3]
@ ARM:   vabd.f16        q0, q1, q2      @ encoding: [0x44,0x0d,0x32,0xf3]
@ THUMB: vabd.f16        d0, d1, d2      @ encoding: [0x31,0xff,0x02,0x0d]
@ THUMB: vabd.f16        q0, q1, q2      @ encoding: [0x32,0xff,0x44,0x0d]

  vabs.f16 d0, d1
  vabs.f16 q0, q1
@ ARM:   vabs.f16        d0, d1          @ encoding: [0x01,0x07,0xb5,0xf3]
@ ARM:   vabs.f16        q0, q1          @ encoding: [0x42,0x07,0xb5,0xf3]
@ THUMB: vabs.f16        d0, d1          @ encoding: [0xb5,0xff,0x01,0x07]
@ THUMB: vabs.f16        q0, q1          @ encoding: [0xb5,0xff,0x42,0x07]

  vmax.f16 d0, d1, d2
  vmax.f16 q0, q1, q2
@ ARM:   vmax.f16        d0, d1, d2      @ encoding: [0x02,0x0f,0x11,0xf2]
@ ARM:   vmax.f16        q0, q1, q2      @ encoding: [0x44,0x0f,0x12,0xf2]
@ THUMB: vmax.f16        d0, d1, d2      @ encoding: [0x11,0xef,0x02,0x0f]
@ THUMB: vmax.f16        q0, q1, q2      @ encoding: [0x12,0xef,0x44,0x0f]

  vmin.f16 d0, d1, d2
  vmin.f16 q0, q1, q2
@ ARM:   vmin.f16        d0, d1, d2      @ encoding: [0x02,0x0f,0x31,0xf2]
@ ARM:   vmin.f16        q0, q1, q2      @ encoding: [0x44,0x0f,0x32,0xf2]
@ THUMB: vmin.f16        d0, d1, d2      @ encoding: [0x31,0xef,0x02,0x0f]
@ THUMB: vmin.f16        q0, q1, q2      @ encoding: [0x32,0xef,0x44,0x0f]

  vmaxnm.f16 d0, d1, d2
  vmaxnm.f16 q0, q1, q2
@ ARM:   vmaxnm.f16      d0, d1, d2      @ encoding: [0x12,0x0f,0x11,0xf3]
@ ARM:   vmaxnm.f16      q0, q1, q2      @ encoding: [0x54,0x0f,0x12,0xf3]
@ THUMB: vmaxnm.f16      d0, d1, d2      @ encoding: [0x11,0xff,0x12,0x0f]
@ THUMB: vmaxnm.f16      q0, q1, q2      @ encoding: [0x12,0xff,0x54,0x0f]

  vminnm.f16 d0, d1, d2
  vminnm.f16 q0, q1, q2
@ ARM:   vminnm.f16      d0, d1, d2      @ encoding: [0x12,0x0f,0x31,0xf3]
@ ARM:   vminnm.f16      q0, q1, q2      @ encoding: [0x54,0x0f,0x32,0xf3]
@ THUMB: vminnm.f16      d0, d1, d2      @ encoding: [0x31,0xff,0x12,0x0f]
@ THUMB: vminnm.f16      q0, q1, q2      @ encoding: [0x32,0xff,0x54,0x0f]

  vpadd.f16 d0, d1, d2
@ ARM:   vpadd.f16       d0, d1, d2      @ encoding: [0x02,0x0d,0x11,0xf3]
@ THUMB: vpadd.f16       d0, d1, d2      @ encoding: [0x11,0xff,0x02,0x0d]

  vpmax.f16 d0, d1, d2
@ ARM:   vpmax.f16       d0, d1, d2      @ encoding: [0x02,0x0f,0x11,0xf3]
@ THUMB: vpmax.f16       d0, d1, d2      @ encoding: [0x11,0xff,0x02,0x0f]

  vpmin.f16 d0, d1, d2
@ ARM:   vpmin.f16       d0, d1, d2      @ encoding: [0x02,0x0f,0x31,0xf3]
@ THUMB: vpmin.f16       d0, d1, d2      @ encoding: [0x31,0xff,0x02,0x0f]

  vrecpe.f16 d0, d1
  vrecpe.f16 q0, q1
@ ARM:   vrecpe.f16      d0, d1          @ encoding: [0x01,0x05,0xb7,0xf3]
@ ARM:   vrecpe.f16      q0, q1          @ encoding: [0x42,0x05,0xb7,0xf3]
@ THUMB: vrecpe.f16      d0, d1          @ encoding: [0xb7,0xff,0x01,0x05]
@ THUMB: vrecpe.f16      q0, q1          @ encoding: [0xb7,0xff,0x42,0x05]

  vrecps.f16 d0, d1, d2
  vrecps.f16 q0, q1, q2
@ ARM:   vrecps.f16      d0, d1, d2      @ encoding: [0x12,0x0f,0x11,0xf2]
@ ARM:   vrecps.f16      q0, q1, q2      @ encoding: [0x54,0x0f,0x12,0xf2]
@ THUMB: vrecps.f16      d0, d1, d2      @ encoding: [0x11,0xef,0x12,0x0f]
@ THUMB: vrecps.f16      q0, q1, q2      @ encoding: [0x12,0xef,0x54,0x0f]

  vrsqrte.f16 d0, d1
  vrsqrte.f16 q0, q1
@ ARM:   vrsqrte.f16     d0, d1          @ encoding: [0x81,0x05,0xb7,0xf3]
@ ARM:   vrsqrte.f16     q0, q1          @ encoding: [0xc2,0x05,0xb7,0xf3]
@ THUMB: vrsqrte.f16     d0, d1          @ encoding: [0xb7,0xff,0x81,0x05]
@ THUMB: vrsqrte.f16     q0, q1          @ encoding: [0xb7,0xff,0xc2,0x05]

  vrsqrts.f16 d0, d1, d2
  vrsqrts.f16 q0, q1, q2
@ ARM:   vrsqrts.f16     d0, d1, d2      @ encoding: [0x12,0x0f,0x31,0xf2]
@ ARM:   vrsqrts.f16     q0, q1, q2      @ encoding: [0x54,0x0f,0x32,0xf2]
@ THUMB: vrsqrts.f16     d0, d1, d2      @ encoding: [0x31,0xef,0x12,0x0f]
@ THUMB: vrsqrts.f16     q0, q1, q2      @ encoding: [0x32,0xef,0x54,0x0f]

  vneg.f16 d0, d1
  vneg.f16 q0, q1
@ ARM:   vneg.f16        d0, d1          @ encoding: [0x81,0x07,0xb5,0xf3]
@ ARM:   vneg.f16        q0, q1          @ encoding: [0xc2,0x07,0xb5,0xf3]
@ THUMB: vneg.f16        d0, d1          @ encoding: [0xb5,0xff,0x81,0x07]
@ THUMB: vneg.f16        q0, q1          @ encoding: [0xb5,0xff,0xc2,0x07]

  vcvt.s16.f16 d0, d1
  vcvt.u16.f16 d0, d1
  vcvt.f16.s16 d0, d1
  vcvt.f16.u16 d0, d1
  vcvt.s16.f16 q0, q1
  vcvt.u16.f16 q0, q1
  vcvt.f16.s16 q0, q1
  vcvt.f16.u16 q0, q1
@ ARM:   vcvt.s16.f16    d0, d1          @ encoding: [0x01,0x07,0xb7,0xf3]
@ ARM:   vcvt.u16.f16    d0, d1          @ encoding: [0x81,0x07,0xb7,0xf3]
@ ARM:   vcvt.f16.s16    d0, d1          @ encoding: [0x01,0x06,0xb7,0xf3]
@ ARM:   vcvt.f16.u16    d0, d1          @ encoding: [0x81,0x06,0xb7,0xf3]
@ ARM:   vcvt.s16.f16    q0, q1          @ encoding: [0x42,0x07,0xb7,0xf3]
@ ARM:   vcvt.u16.f16    q0, q1          @ encoding: [0xc2,0x07,0xb7,0xf3]
@ ARM:   vcvt.f16.s16    q0, q1          @ encoding: [0x42,0x06,0xb7,0xf3]
@ ARM:   vcvt.f16.u16    q0, q1          @ encoding: [0xc2,0x06,0xb7,0xf3]
@ THUMB: vcvt.s16.f16    d0, d1          @ encoding: [0xb7,0xff,0x01,0x07]
@ THUMB: vcvt.u16.f16    d0, d1          @ encoding: [0xb7,0xff,0x81,0x07]
@ THUMB: vcvt.f16.s16    d0, d1          @ encoding: [0xb7,0xff,0x01,0x06]
@ THUMB: vcvt.f16.u16    d0, d1          @ encoding: [0xb7,0xff,0x81,0x06]
@ THUMB: vcvt.s16.f16    q0, q1          @ encoding: [0xb7,0xff,0x42,0x07]
@ THUMB: vcvt.u16.f16    q0, q1          @ encoding: [0xb7,0xff,0xc2,0x07]
@ THUMB: vcvt.f16.s16    q0, q1          @ encoding: [0xb7,0xff,0x42,0x06]
@ THUMB: vcvt.f16.u16    q0, q1          @ encoding: [0xb7,0xff,0xc2,0x06]

  vcvta.s16.f16 d0, d1
  vcvta.s16.f16 q0, q1
  vcvta.u16.f16 d0, d1
  vcvta.u16.f16 q0, q1
@ ARM:   vcvta.s16.f16   d0, d1          @ encoding: [0x01,0x00,0xb7,0xf3]
@ ARM:   vcvta.s16.f16   q0, q1          @ encoding: [0x42,0x00,0xb7,0xf3]
@ ARM:   vcvta.u16.f16   d0, d1          @ encoding: [0x81,0x00,0xb7,0xf3]
@ ARM:   vcvta.u16.f16   q0, q1          @ encoding: [0xc2,0x00,0xb7,0xf3]
@ THUMB: vcvta.s16.f16   d0, d1          @ encoding: [0xb7,0xff,0x01,0x00]
@ THUMB: vcvta.s16.f16   q0, q1          @ encoding: [0xb7,0xff,0x42,0x00]
@ THUMB: vcvta.u16.f16   d0, d1          @ encoding: [0xb7,0xff,0x81,0x00]
@ THUMB: vcvta.u16.f16   q0, q1          @ encoding: [0xb7,0xff,0xc2,0x00]

  vcvtm.s16.f16 d0, d1
  vcvtm.s16.f16 q0, q1
  vcvtm.u16.f16 d0, d1
  vcvtm.u16.f16 q0, q1
@ ARM:   vcvtm.s16.f16   d0, d1          @ encoding: [0x01,0x03,0xb7,0xf3]
@ ARM:   vcvtm.s16.f16   q0, q1          @ encoding: [0x42,0x03,0xb7,0xf3]
@ ARM:   vcvtm.u16.f16   d0, d1          @ encoding: [0x81,0x03,0xb7,0xf3]
@ ARM:   vcvtm.u16.f16   q0, q1          @ encoding: [0xc2,0x03,0xb7,0xf3]
@ THUMB: vcvtm.s16.f16   d0, d1          @ encoding: [0xb7,0xff,0x01,0x03]
@ THUMB: vcvtm.s16.f16   q0, q1          @ encoding: [0xb7,0xff,0x42,0x03]
@ THUMB: vcvtm.u16.f16   d0, d1          @ encoding: [0xb7,0xff,0x81,0x03]
@ THUMB: vcvtm.u16.f16   q0, q1          @ encoding: [0xb7,0xff,0xc2,0x03]

  vcvtn.s16.f16 d0, d1
  vcvtn.s16.f16 q0, q1
  vcvtn.u16.f16 d0, d1
  vcvtn.u16.f16 q0, q1
@ ARM:   vcvtn.s16.f16   d0, d1          @ encoding: [0x01,0x01,0xb7,0xf3]
@ ARM:   vcvtn.s16.f16   q0, q1          @ encoding: [0x42,0x01,0xb7,0xf3]
@ ARM:   vcvtn.u16.f16   d0, d1          @ encoding: [0x81,0x01,0xb7,0xf3]
@ ARM:   vcvtn.u16.f16   q0, q1          @ encoding: [0xc2,0x01,0xb7,0xf3]
@ THUMB: vcvtn.s16.f16   d0, d1          @ encoding: [0xb7,0xff,0x01,0x01]
@ THUMB: vcvtn.s16.f16   q0, q1          @ encoding: [0xb7,0xff,0x42,0x01]
@ THUMB: vcvtn.u16.f16   d0, d1          @ encoding: [0xb7,0xff,0x81,0x01]
@ THUMB: vcvtn.u16.f16   q0, q1          @ encoding: [0xb7,0xff,0xc2,0x01]

  vcvtp.s16.f16 d0, d1
  vcvtp.s16.f16 q0, q1
  vcvtp.u16.f16 d0, d1
  vcvtp.u16.f16 q0, q1
@ ARM:   vcvtp.s16.f16   d0, d1          @ encoding: [0x01,0x02,0xb7,0xf3]
@ ARM:   vcvtp.s16.f16   q0, q1          @ encoding: [0x42,0x02,0xb7,0xf3]
@ ARM:   vcvtp.u16.f16   d0, d1          @ encoding: [0x81,0x02,0xb7,0xf3]
@ ARM:   vcvtp.u16.f16   q0, q1          @ encoding: [0xc2,0x02,0xb7,0xf3]
@ THUMB: vcvtp.s16.f16   d0, d1          @ encoding: [0xb7,0xff,0x01,0x02]
@ THUMB: vcvtp.s16.f16   q0, q1          @ encoding: [0xb7,0xff,0x42,0x02]
@ THUMB: vcvtp.u16.f16   d0, d1          @ encoding: [0xb7,0xff,0x81,0x02]
@ THUMB: vcvtp.u16.f16   q0, q1          @ encoding: [0xb7,0xff,0xc2,0x02]


  vcvt.s16.f16 d0, d1, #1
  vcvt.u16.f16 d0, d1, #2
  vcvt.f16.s16 d0, d1, #3
  vcvt.f16.u16 d0, d1, #4
  vcvt.s16.f16 q0, q1, #5
  vcvt.u16.f16 q0, q1, #6
  vcvt.f16.s16 q0, q1, #7
  vcvt.f16.u16 q0, q1, #8
@ ARM:   vcvt.s16.f16    d0, d1, #1      @ encoding: [0x11,0x0d,0xbf,0xf2]
@ ARM:   vcvt.u16.f16    d0, d1, #2      @ encoding: [0x11,0x0d,0xbe,0xf3]
@ ARM:   vcvt.f16.s16    d0, d1, #3      @ encoding: [0x11,0x0c,0xbd,0xf2]
@ ARM:   vcvt.f16.u16    d0, d1, #4      @ encoding: [0x11,0x0c,0xbc,0xf3]
@ ARM:   vcvt.s16.f16    q0, q1, #5      @ encoding: [0x52,0x0d,0xbb,0xf2]
@ ARM:   vcvt.u16.f16    q0, q1, #6      @ encoding: [0x52,0x0d,0xba,0xf3]
@ ARM:   vcvt.f16.s16    q0, q1, #7      @ encoding: [0x52,0x0c,0xb9,0xf2]
@ ARM:   vcvt.f16.u16    q0, q1, #8      @ encoding: [0x52,0x0c,0xb8,0xf3]
@ THUMB: vcvt.s16.f16    d0, d1, #1      @ encoding: [0xbf,0xef,0x11,0x0d]
@ THUMB: vcvt.u16.f16    d0, d1, #2      @ encoding: [0xbe,0xff,0x11,0x0d]
@ THUMB: vcvt.f16.s16    d0, d1, #3      @ encoding: [0xbd,0xef,0x11,0x0c]
@ THUMB: vcvt.f16.u16    d0, d1, #4      @ encoding: [0xbc,0xff,0x11,0x0c]
@ THUMB: vcvt.s16.f16    q0, q1, #5      @ encoding: [0xbb,0xef,0x52,0x0d]
@ THUMB: vcvt.u16.f16    q0, q1, #6      @ encoding: [0xba,0xff,0x52,0x0d]
@ THUMB: vcvt.f16.s16    q0, q1, #7      @ encoding: [0xb9,0xef,0x52,0x0c]
@ THUMB: vcvt.f16.u16    q0, q1, #8      @ encoding: [0xb8,0xff,0x52,0x0c]

  vrinta.f16.f16 d0, d1
  vrinta.f16.f16 q0, q1
@ ARM:   vrinta.f16      d0, d1          @ encoding: [0x01,0x05,0xb6,0xf3]
@ ARM:   vrinta.f16      q0, q1          @ encoding: [0x42,0x05,0xb6,0xf3]
@ THUMB: vrinta.f16      d0, d1          @ encoding: [0xb6,0xff,0x01,0x05]
@ THUMB: vrinta.f16      q0, q1          @ encoding: [0xb6,0xff,0x42,0x05]

  vrintm.f16.f16 d0, d1
  vrintm.f16.f16 q0, q1
@ ARM:   vrintm.f16      d0, d1          @ encoding: [0x81,0x06,0xb6,0xf3]
@ ARM:   vrintm.f16      q0, q1          @ encoding: [0xc2,0x06,0xb6,0xf3]
@ THUMB: vrintm.f16      d0, d1          @ encoding: [0xb6,0xff,0x81,0x06]
@ THUMB: vrintm.f16      q0, q1          @ encoding: [0xb6,0xff,0xc2,0x06]

  vrintn.f16.f16 d0, d1
  vrintn.f16.f16 q0, q1
@ ARM:   vrintn.f16      d0, d1          @ encoding: [0x01,0x04,0xb6,0xf3]
@ ARM:   vrintn.f16      q0, q1          @ encoding: [0x42,0x04,0xb6,0xf3]
@ THUMB: vrintn.f16      d0, d1          @ encoding: [0xb6,0xff,0x01,0x04]
@ THUMB: vrintn.f16      q0, q1          @ encoding: [0xb6,0xff,0x42,0x04]

  vrintp.f16.f16 d0, d1
  vrintp.f16.f16 q0, q1
@ ARM:   vrintp.f16      d0, d1          @ encoding: [0x81,0x07,0xb6,0xf3]
@ ARM:   vrintp.f16      q0, q1          @ encoding: [0xc2,0x07,0xb6,0xf3]
@ THUMB: vrintp.f16      d0, d1          @ encoding: [0xb6,0xff,0x81,0x07]
@ THUMB: vrintp.f16      q0, q1          @ encoding: [0xb6,0xff,0xc2,0x07]

  vrintx.f16.f16 d0, d1
  vrintx.f16.f16 q0, q1
@ ARM:   vrintx.f16      d0, d1          @ encoding: [0x81,0x04,0xb6,0xf3]
@ ARM:   vrintx.f16      q0, q1          @ encoding: [0xc2,0x04,0xb6,0xf3]
@ THUMB: vrintx.f16      d0, d1          @ encoding: [0xb6,0xff,0x81,0x04]
@ THUMB: vrintx.f16      q0, q1          @ encoding: [0xb6,0xff,0xc2,0x04]

  vrintz.f16.f16 d0, d1
  vrintz.f16.f16 q0, q1
@ ARM:   vrintz.f16      d0, d1          @ encoding: [0x81,0x05,0xb6,0xf3]
@ ARM:   vrintz.f16      q0, q1          @ encoding: [0xc2,0x05,0xb6,0xf3]
@ THUMB: vrintz.f16      d0, d1          @ encoding: [0xb6,0xff,0x81,0x05]
@ THUMB: vrintz.f16      q0, q1          @ encoding: [0xb6,0xff,0xc2,0x05]
