@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16,+neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16,-neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8a-none-eabi -mattr=-fullfp16,+neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8a-none-eabi -mattr=+fullfp16,-neon -show-encoding < %s 2>&1 | FileCheck %s

  vadd.f16 d0, d1, d2
  vadd.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vsub.f16 d0, d1, d2
  vsub.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmul.f16 d0, d1, d2
  vmul.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmul.f16 d1, d2, d3[2]
  vmul.f16 q4, q5, d6[3]
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmla.f16 d0, d1, d2
  vmla.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmla.f16 d5, d6, d7[2]
  vmla.f16 q5, q6, d7[3]
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmls.f16 d0, d1, d2
  vmls.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmls.f16 d5, d6, d7[2]
  vmls.f16 q5, q6, d7[3]
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vfma.f16 d0, d1, d2
  vfma.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vfms.f16 d0, d1, d2
  vfms.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vceq.f16 d2, d3, d4
  vceq.f16 q2, q3, q4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vceq.f16 d2, d3, #0
  vceq.f16 q2, q3, #0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcge.f16 d2, d3, d4
  vcge.f16 q2, q3, q4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcge.f16 d2, d3, #0
  vcge.f16 q2, q3, #0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcgt.f16 d2, d3, d4
  vcgt.f16 q2, q3, q4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcgt.f16 d2, d3, #0
  vcgt.f16 q2, q3, #0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcle.f16 d2, d3, d4
  vcle.f16 q2, q3, q4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcle.f16 d2, d3, #0
  vcle.f16 q2, q3, #0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vclt.f16 d2, d3, d4
  vclt.f16 q2, q3, q4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vclt.f16 d2, d3, #0
  vclt.f16 q2, q3, #0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vacge.f16 d0, d1, d2
  vacge.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vacgt.f16 d0, d1, d2
  vacgt.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vacle.f16 d0, d1, d2
  vacle.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vaclt.f16 d0, d1, d2
  vaclt.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vabd.f16 d0, d1, d2
  vabd.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vabs.f16 d0, d1
  vabs.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmax.f16 d0, d1, d2
  vmax.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmin.f16 d0, d1, d2
  vmin.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmaxnm.f16 d0, d1, d2
  vmaxnm.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vminnm.f16 d0, d1, d2
  vminnm.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vpadd.f16 d0, d1, d2
@ CHECK: error: instruction requires:

  vpmax.f16 d0, d1, d2
@ CHECK: error: instruction requires:

  vpmin.f16 d0, d1, d2
@ CHECK: error: instruction requires:

  vrecpe.f16 d0, d1
  vrecpe.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrecps.f16 d0, d1, d2
  vrecps.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrsqrte.f16 d0, d1
  vrsqrte.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrsqrts.f16 d0, d1, d2
  vrsqrts.f16 q0, q1, q2
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vneg.f16 d0, d1
  vneg.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvt.s16.f16 d0, d1
  vcvt.u16.f16 d0, d1
  vcvt.f16.s16 d0, d1
  vcvt.f16.u16 d0, d1
  vcvt.s16.f16 q0, q1
  vcvt.u16.f16 q0, q1
  vcvt.f16.s16 q0, q1
  vcvt.f16.u16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvta.s16.f16 d0, d1
  vcvta.s16.f16 q0, q1
  vcvta.u16.f16 d0, d1
  vcvta.u16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvtm.s16.f16 d0, d1
  vcvtm.s16.f16 q0, q1
  vcvtm.u16.f16 d0, d1
  vcvtm.u16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvtn.s16.f16 d0, d1
  vcvtn.s16.f16 q0, q1
  vcvtn.u16.f16 d0, d1
  vcvtn.u16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvtp.s16.f16 d0, d1
  vcvtp.s16.f16 q0, q1
  vcvtp.u16.f16 d0, d1
  vcvtp.u16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:


  vcvt.s16.f16 d0, d1, #1
  vcvt.u16.f16 d0, d1, #2
  vcvt.f16.s16 d0, d1, #3
  vcvt.f16.u16 d0, d1, #4
  vcvt.s16.f16 q0, q1, #5
  vcvt.u16.f16 q0, q1, #6
  vcvt.f16.s16 q0, q1, #7
  vcvt.f16.u16 q0, q1, #8
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrinta.f16.f16 d0, d1
  vrinta.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrintm.f16.f16 d0, d1
  vrintm.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrintn.f16.f16 d0, d1
  vrintn.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrintp.f16.f16 d0, d1
  vrintp.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrintx.f16.f16 d0, d1
  vrintx.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vrintz.f16.f16 d0, d1
  vrintz.f16.f16 q0, q1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
