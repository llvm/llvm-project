@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16 -show-encoding < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=-fullfp16,+thumb-mode -show-encoding < %s 2>&1 | FileCheck %s

         vadd.f16  s0, s1, s0
@ CHECK: error: instruction requires:

         vsub.f16  s0, s1, s0
@ CHECK: error: instruction requires:

         vdiv.f16  s0, s1, s0
@ CHECK: error: instruction requires:

         vmul.f16  s0, s1, s0
@ CHECK: error: instruction requires:

         vnmul.f16       s0, s1, s0
@ CHECK: error: instruction requires:

         vmla.f16        s1, s2, s0
@ CHECK: error: instruction requires:

         vmls.f16        s1, s2, s0
@ CHECK: error: instruction requires:

         vnmla.f16       s1, s2, s0
@ CHECK: error: instruction requires:

         vnmls.f16       s1, s2, s0
@ CHECK: error: instruction requires:

         vcmp.f16 s0, s1
@ CHECK: error: instruction requires:

         vcmp.f16 s2, #0
@ CHECK: error: instruction requires:

         vcmpe.f16       s1, s0
@ CHECK: error: instruction requires:

         vcmpe.f16       s0, #0
@ CHECK: error: instruction requires:

         vabs.f16        s0, s0
@ CHECK: error: instruction requires:

         vneg.f16        s0, s0
@ CHECK: error: instruction requires:

         vsqrt.f16       s0, s0
@ CHECK: error: instruction requires:

         vcvt.f16.s32    s0, s0
         vcvt.f16.u32    s0, s0
         vcvt.s32.f16    s0, s0
         vcvt.u32.f16    s0, s0
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

         vcvtr.s32.f16  s0, s1
         vcvtr.u32.f16  s0, s1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

         vcvt.f16.u32 s0, s0, #20
         vcvt.f16.u16 s0, s0, #1
         vcvt.f16.s32 s1, s1, #20
         vcvt.f16.s16 s17, s17, #1
         vcvt.u32.f16 s12, s12, #20
         vcvt.u16.f16 s28, s28, #1
         vcvt.s32.f16 s1, s1, #20
         vcvt.s16.f16 s17, s17, #1
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vcvta.s32.f16 s2, s3
@ CHECK: error: instruction requires:

  vcvtn.s32.f16 s6, s23
@ CHECK: error: instruction requires:

  vcvtp.s32.f16 s0, s4
@ CHECK: error: instruction requires:

  vcvtm.s32.f16 s17, s8
@ CHECK: error: instruction requires:

  vcvta.u32.f16 s2, s3
@ CHECK: error: instruction requires:

  vcvtn.u32.f16 s6, s23
@ CHECK: error: instruction requires:

  vcvtp.u32.f16 s0, s4
@ CHECK: error: instruction requires:

  vcvtm.u32.f16 s17, s8
@ CHECK: error: instruction requires:

  vselge.f16 s4, s1, s23
@ CHECK: error: instruction requires:

  vselgt.f16 s0, s1, s0
@ CHECK: error: instruction requires:

  vseleq.f16 s30, s28, s23
@ CHECK: error: instruction requires:

  vselvs.f16 s21, s16, s14
@ CHECK: error: instruction requires:

  vmaxnm.f16 s5, s12, s0
@ CHECK: error: instruction requires:

  vminnm.f16 s0, s0, s12
@ CHECK: error: instruction requires:

  vrintz.f16 s3, s24
@ CHECK: error: instruction requires:

  vrintr.f16 s0, s9
@ CHECK: error: instruction requires:

  vrintx.f16 s10, s14
@ CHECK: error: instruction requires:

  vrinta.f16 s12, s1
@ CHECK: error: instruction requires:

  vrintn.f16 s12, s1
@ CHECK: error: instruction requires:

  vrintp.f16 s12, s1
@ CHECK: error: instruction requires:

  vrintm.f16 s12, s1
@ CHECK: error: instruction requires:

  vfma.f16 s2, s7, s4
@ CHECK: error: instruction requires:

  vfms.f16 s2, s7, s4
@ CHECK: error: instruction requires:

  vfnma.f16 s2, s7, s4
@ CHECK: error: instruction requires:

  vfnms.f16 s2, s7, s4
@ CHECK: error: instruction requires:

  vmovx.f16 s2, s5
  vins.f16 s2, s5
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:


  vldr.16 s1, [pc, #6]
  vldr.16 s2, [pc, #510]
  vldr.16 s3, [pc, #-510]
  vldr.16 s4, [r4, #-18]
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:


  vstr.16 s1, [pc, #6]
  vstr.16 s2, [pc, #510]
  vstr.16 s3, [pc, #-510]
  vstr.16 s4, [r4, #-18]
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:

  vmov.f16 s0, #1.0
@ CHECK: error: instruction requires:

  vmov.f16 s1, r2
  vmov.f16 r3, s4
@ CHECK: error: instruction requires:
@ CHECK: error: instruction requires:
