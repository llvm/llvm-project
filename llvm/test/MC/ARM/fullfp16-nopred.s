@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16 < %s 2>&1 | FileCheck %s
@ RUN: not llvm-mc -triple armv8a-none-eabi -mattr=+fullfp16,+thumb-mode -arm-implicit-it always < %s 2>&1 | FileCheck %s

  vaddeq.f16  s0, s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vsubne.f16  s0, s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vdivmi.f16  s0, s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmulpl.f16  s0, s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vnmulvs.f16       s0, s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmlavc.f16        s1, s2, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmlshs.f16        s1, s2, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vnmlalo.f16       s1, s2, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vnmlscs.f16       s1, s2, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcmpcc.f16 s0, s1
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcmphi.f16 s2, #0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcmpels.f16       s1, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcmpege.f16       s0, #0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vabslt.f16        s0, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vneggt.f16        s0, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vsqrtle.f16       s0, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcvteq.f16.s32    s0, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcvtne.u32.f16    s0, s0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vcvtrmi.s32.f16  s0, s1
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vrintzhs.f16 s3, s24
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vrintrlo.f16 s0, s9
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vrintxcs.f16 s10, s14
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vfmalt.f16 s2, s7, s4
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vfmsgt.f16 s2, s7, s4
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vfnmale.f16 s2, s7, s4
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vfnmseq.f16 s2, s7, s4
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vldrpl.16 s1, [pc, #6]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vldrvs.16 s2, [pc, #510]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vldrvc.16 s3, [pc, #-510]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vldrhs.16 s4, [r4, #-18]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vstrlo.16 s1, [pc, #6]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vstrcs.16 s2, [pc, #510]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vstrcc.16 s3, [pc, #-510]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vstrhi.16 s4, [r4, #-18]
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmovls.f16 s0, #1.0
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmovge.f16 s1, r2
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable

  vmovlt.f16 r3, s4
@ CHECK: [[@LINE-1]]:3: error: instruction is not predicable
