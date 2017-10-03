@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16,+neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16,-neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8a-none-eabi -mattr=-fullfp16,+neon -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple thumbv8a-none-eabi -mattr=+fullfp16,-neon -show-encoding < %s 2>&1 | FileCheck %s

  vadd.f16 d0, d1, d2
  vadd.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vsub.f16 d0, d1, d2
  vsub.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmul.f16 d0, d1, d2
  vmul.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmul.f16 d1, d2, d3[2]
  vmul.f16 q4, q5, d6[3]
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmla.f16 d0, d1, d2
  vmla.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmla.f16 d5, d6, d7[2]
  vmla.f16 q5, q6, d7[3]
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmls.f16 d0, d1, d2
  vmls.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmls.f16 d5, d6, d7[2]
  vmls.f16 q5, q6, d7[3]
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vfma.f16 d0, d1, d2
  vfma.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vfms.f16 d0, d1, d2
  vfms.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vceq.f16 d2, d3, d4
  vceq.f16 q2, q3, q4
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vceq.f16 d2, d3, #0
  vceq.f16 q2, q3, #0
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcge.f16 d2, d3, d4
  vcge.f16 q2, q3, q4
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcge.f16 d2, d3, #0
  vcge.f16 q2, q3, #0
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcgt.f16 d2, d3, d4
  vcgt.f16 q2, q3, q4
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcgt.f16 d2, d3, #0
  vcgt.f16 q2, q3, #0
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcle.f16 d2, d3, d4
  vcle.f16 q2, q3, q4
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcle.f16 d2, d3, #0
  vcle.f16 q2, q3, #0
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vclt.f16 d2, d3, d4
  vclt.f16 q2, q3, q4
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vclt.f16 d2, d3, #0
  vclt.f16 q2, q3, #0
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vacge.f16 d0, d1, d2
  vacge.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vacgt.f16 d0, d1, d2
  vacgt.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vacle.f16 d0, d1, d2
  vacle.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vaclt.f16 d0, d1, d2
  vaclt.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vabd.f16 d0, d1, d2
  vabd.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vabs.f16 d0, d1
  vabs.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmax.f16 d0, d1, d2
  vmax.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmin.f16 d0, d1, d2
  vmin.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vmaxnm.f16 d0, d1, d2
  vmaxnm.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vminnm.f16 d0, d1, d2
  vminnm.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vpadd.f16 d0, d1, d2
@ CHECK: instruction requires: {{full half-float|NEON}}

  vpmax.f16 d0, d1, d2
@ CHECK: instruction requires: {{full half-float|NEON}}

  vpmin.f16 d0, d1, d2
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrecpe.f16 d0, d1
  vrecpe.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrecps.f16 d0, d1, d2
  vrecps.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrsqrte.f16 d0, d1
  vrsqrte.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrsqrts.f16 d0, d1, d2
  vrsqrts.f16 q0, q1, q2
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vneg.f16 d0, d1
  vneg.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcvt.s16.f16 d0, d1
  vcvt.u16.f16 d0, d1
  vcvt.f16.s16 d0, d1
  vcvt.f16.u16 d0, d1
  vcvt.s16.f16 q0, q1
  vcvt.u16.f16 q0, q1
  vcvt.f16.s16 q0, q1
  vcvt.f16.u16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcvta.s16.f16 d0, d1
  vcvta.s16.f16 q0, q1
  vcvta.u16.f16 d0, d1
  vcvta.u16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcvtm.s16.f16 d0, d1
  vcvtm.s16.f16 q0, q1
  vcvtm.u16.f16 d0, d1
  vcvtm.u16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcvtn.s16.f16 d0, d1
  vcvtn.s16.f16 q0, q1
  vcvtn.u16.f16 d0, d1
  vcvtn.u16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vcvtp.s16.f16 d0, d1
  vcvtp.s16.f16 q0, q1
  vcvtp.u16.f16 d0, d1
  vcvtp.u16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}


  vcvt.s16.f16 d0, d1, #1
  vcvt.u16.f16 d0, d1, #2
  vcvt.f16.s16 d0, d1, #3
  vcvt.f16.u16 d0, d1, #4
  vcvt.s16.f16 q0, q1, #5
  vcvt.u16.f16 q0, q1, #6
  vcvt.f16.s16 q0, q1, #7
  vcvt.f16.u16 q0, q1, #8
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrinta.f16.f16 d0, d1
  vrinta.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrintm.f16.f16 d0, d1
  vrintm.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrintn.f16.f16 d0, d1
  vrintn.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrintp.f16.f16 d0, d1
  vrintp.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrintx.f16.f16 d0, d1
  vrintx.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}

  vrintz.f16.f16 d0, d1
  vrintz.f16.f16 q0, q1
@ CHECK: instruction requires: {{full half-float|NEON}}
@ CHECK: instruction requires: {{full half-float|NEON}}
