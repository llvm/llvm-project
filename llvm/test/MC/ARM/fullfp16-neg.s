@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16 -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16,+thumb-mode -show-encoding < %s 2>&1 | FileCheck %s

         vadd.f16  s0, s1, s0
@ CHECK: instruction requires: full half-float

         vsub.f16  s0, s1, s0
@ CHECK: instruction requires: full half-float

         vdiv.f16  s0, s1, s0
@ CHECK: instruction requires: full half-float

         vmul.f16  s0, s1, s0
@ CHECK: instruction requires: full half-float

         vnmul.f16       s0, s1, s0
@ CHECK: instruction requires: full half-float

         vmla.f16        s1, s2, s0
@ CHECK: instruction requires: full half-float

         vmls.f16        s1, s2, s0
@ CHECK: instruction requires: full half-float

         vnmla.f16       s1, s2, s0
@ CHECK: instruction requires: full half-float

         vnmls.f16       s1, s2, s0
@ CHECK: instruction requires: full half-float

         vcmp.f16 s0, s1
@ CHECK: instruction requires: full half-float

         vcmp.f16 s2, #0
@ CHECK: instruction requires: full half-float

         vcmpe.f16       s1, s0
@ CHECK: instruction requires: full half-float

         vcmpe.f16       s0, #0
@ CHECK: instruction requires: full half-float

         vabs.f16        s0, s0
@ CHECK: instruction requires: full half-float

         vneg.f16        s0, s0
@ CHECK: instruction requires: full half-float

         vsqrt.f16       s0, s0
@ CHECK: instruction requires: full half-float

         vcvt.f16.s32    s0, s0
         vcvt.f16.u32    s0, s0
         vcvt.s32.f16    s0, s0
         vcvt.u32.f16    s0, s0
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

         vcvtr.s32.f16  s0, s1
         vcvtr.u32.f16  s0, s1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

         vcvt.f16.u32 s0, s0, #20
         vcvt.f16.u16 s0, s0, #1
         vcvt.f16.s32 s1, s1, #20
         vcvt.f16.s16 s17, s17, #1
         vcvt.u32.f16 s12, s12, #20
         vcvt.u16.f16 s28, s28, #1
         vcvt.s32.f16 s1, s1, #20
         vcvt.s16.f16 s17, s17, #1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vcvta.s32.f16 s2, s3
@ CHECK: instruction requires: full half-float

  vcvtn.s32.f16 s6, s23
@ CHECK: instruction requires: full half-float

  vcvtp.s32.f16 s0, s4
@ CHECK: instruction requires: full half-float

  vcvtm.s32.f16 s17, s8
@ CHECK: instruction requires: full half-float

  vcvta.u32.f16 s2, s3
@ CHECK: instruction requires: full half-float

  vcvtn.u32.f16 s6, s23
@ CHECK: instruction requires: full half-float

  vcvtp.u32.f16 s0, s4
@ CHECK: instruction requires: full half-float

  vcvtm.u32.f16 s17, s8
@ CHECK: instruction requires: full half-float

  vselge.f16 s4, s1, s23
@ CHECK: instruction requires: full half-float

  vselgt.f16 s0, s1, s0
@ CHECK: instruction requires: full half-float

  vseleq.f16 s30, s28, s23
@ CHECK: instruction requires: full half-float

  vselvs.f16 s21, s16, s14
@ CHECK: instruction requires: full half-float

  vmaxnm.f16 s5, s12, s0
@ CHECK: instruction requires: full half-float

  vminnm.f16 s0, s0, s12
@ CHECK: instruction requires: full half-float

  vrintz.f16 s3, s24
  vrintz.f16.f16 s3, s24
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrintr.f16 s0, s9
  vrintr.f16.f16 s0, s9
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrintx.f16 s10, s14
  vrintx.f16.f16 s10, s14
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrinta.f16 s12, s1
  vrinta.f16.f16 s12, s1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrintn.f16 s12, s1
  vrintn.f16.f16 s12, s1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrintp.f16 s12, s1
  vrintp.f16.f16 s12, s1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vrintm.f16 s12, s1
  vrintm.f16.f16 s12, s1
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float

  vfma.f16 s2, s7, s4
@ CHECK: instruction requires: full half-float

  vfms.f16 s2, s7, s4
@ CHECK: instruction requires: full half-float

  vfnma.f16 s2, s7, s4
@ CHECK: instruction requires: full half-float

  vfnms.f16 s2, s7, s4
@ CHECK: instruction requires: full half-float

  vmovx.f16 s2, s5
  vins.f16 s2, s5
@ CHECK: instruction requires: full half-float
@ CHECK: instruction requires: full half-float


  vldr.16 s1, [pc, #6]
  vldr.16 s2, [pc, #510]
  vldr.16 s3, [pc, #-510]
  vldr.16 s4, [r4, #-18]
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers


  vstr.16 s1, [pc, #6]
  vstr.16 s2, [pc, #510]
  vstr.16 s3, [pc, #-510]
  vstr.16 s4, [r4, #-18]
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers

  vmov.f16 s0, #1.0
@ CHECK: instruction requires: full half-float

  vmov.f16 s1, r2
  vmov.f16 r3, s4
@ CHECK: instruction requires: 16-bit fp registers
@ CHECK: instruction requires: 16-bit fp registers
