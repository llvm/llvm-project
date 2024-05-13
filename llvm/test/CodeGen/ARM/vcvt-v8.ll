; RUN: llc < %s -mtriple=armv8 -mattr=+neon | FileCheck %s
define <4 x i32> @vcvtasq(ptr %A) {
; CHECK: vcvtasq
; CHECK: vcvta.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtas.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtasd(ptr %A) {
; CHECK: vcvtasd
; CHECK: vcvta.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtas.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtnsq(ptr %A) {
; CHECK: vcvtnsq
; CHECK: vcvtn.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtns.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtnsd(ptr %A) {
; CHECK: vcvtnsd
; CHECK: vcvtn.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtns.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtpsq(ptr %A) {
; CHECK: vcvtpsq
; CHECK: vcvtp.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtps.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtpsd(ptr %A) {
; CHECK: vcvtpsd
; CHECK: vcvtp.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtps.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtmsq(ptr %A) {
; CHECK: vcvtmsq
; CHECK: vcvtm.s32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtms.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtmsd(ptr %A) {
; CHECK: vcvtmsd
; CHECK: vcvtm.s32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtms.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtauq(ptr %A) {
; CHECK: vcvtauq
; CHECK: vcvta.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtau.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtaud(ptr %A) {
; CHECK: vcvtaud
; CHECK: vcvta.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtau.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtnuq(ptr %A) {
; CHECK: vcvtnuq
; CHECK: vcvtn.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtnu.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtnud(ptr %A) {
; CHECK: vcvtnud
; CHECK: vcvtn.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtnu.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtpuq(ptr %A) {
; CHECK: vcvtpuq
; CHECK: vcvtp.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtpu.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtpud(ptr %A) {
; CHECK: vcvtpud
; CHECK: vcvtp.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtpu.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

define <4 x i32> @vcvtmuq(ptr %A) {
; CHECK: vcvtmuq
; CHECK: vcvtm.u32.f32 q{{[0-9]+}}, q{{[0-9]+}}
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = call <4 x i32> @llvm.arm.neon.vcvtmu.v4i32.v4f32(<4 x float> %tmp1)
  ret <4 x i32> %tmp2
}

define <2 x i32> @vcvtmud(ptr %A) {
; CHECK: vcvtmud
; CHECK: vcvtm.u32.f32 d{{[0-9]+}}, d{{[0-9]+}}
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = call <2 x i32> @llvm.arm.neon.vcvtmu.v2i32.v2f32(<2 x float> %tmp1)
  ret <2 x i32> %tmp2
}

declare <4 x i32> @llvm.arm.neon.vcvtas.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtas.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtns.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtns.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtps.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtps.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtms.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtms.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtau.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtau.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtnu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtnu.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtpu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtpu.v2i32.v2f32(<2 x float>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vcvtmu.v4i32.v4f32(<4 x float>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vcvtmu.v2i32.v2f32(<2 x float>) nounwind readnone
