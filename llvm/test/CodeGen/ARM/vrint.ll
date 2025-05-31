; RUN: llc -mtriple=armv8 -mattr=+neon %s -o - | FileCheck %s

declare float @llvm.arm.neon.vrintn.f32(float) nounwind readnone
declare <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float>) nounwind readnone

; CHECK-LABEL: vrintn_f32:
; CHECK: vrintn.f32
define float @vrintn_f32(ptr %A) nounwind {
  %tmp1 = load float, ptr %A
  %tmp2 = call float @llvm.arm.neon.vrintn.f32(float %tmp1)
  ret float %tmp2
}

define <2 x float> @frintn_2s(<2 x float> %A) nounwind {
; CHECK-LABEL: frintn_2s:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d16, r0, r1
; CHECK-NEXT:    vrintn.f32 d16, d16
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    bx lr
	%tmp3 = call <2 x float> @llvm.arm.neon.vrintn.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @frintn_4s(<4 x float> %A) nounwind {
; CHECK-LABEL: frintn_4s:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d17, r2, r3
; CHECK-NEXT:    vmov d16, r0, r1
; CHECK-NEXT:    vrintn.f32 q8, q8
; CHECK-NEXT:    vmov r0, r1, d16
; CHECK-NEXT:    vmov r2, r3, d17
; CHECK-NEXT:    bx lr
	%tmp3 = call <4 x float> @llvm.arm.neon.vrintn.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <4 x half> @roundeven_4h(<4 x half> %A) nounwind {
; CHECK-LABEL: roundeven_4h:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov s0, r3
; CHECK-NEXT:    vcvtb.f32.f16 s0, s0
; CHECK-NEXT:    vmov s2, r2
; CHECK-NEXT:    vrintn.f32 s0, s0
; CHECK-NEXT:    vcvtb.f32.f16 s2, s2
; CHECK-NEXT:    vrintn.f32 s2, s2
; CHECK-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-NEXT:    vcvtb.f16.f32 s2, s2
; CHECK-NEXT:    vmov r2, s0
; CHECK-NEXT:    vmov s0, r1
; CHECK-NEXT:    vmov r3, s2
; CHECK-NEXT:    vcvtb.f32.f16 s0, s0
; CHECK-NEXT:    vmov s2, r0
; CHECK-NEXT:    vrintn.f32 s0, s0
; CHECK-NEXT:    vcvtb.f32.f16 s2, s2
; CHECK-NEXT:    vcvtb.f16.f32 s0, s0
; CHECK-NEXT:    vrintn.f32 s2, s2
; CHECK-NEXT:    vmov r0, s0
; CHECK-NEXT:    vcvtb.f16.f32 s2, s2
; CHECK-NEXT:    vmov r1, s2
; CHECK-NEXT:    pkhbt r2, r3, r2, lsl #16
; CHECK-NEXT:    pkhbt r0, r1, r0, lsl #16
; CHECK-NEXT:    vmov d16, r0, r2
; CHECK-NEXT:    vmov.u16 r0, d16[0]
; CHECK-NEXT:    vmov.u16 r1, d16[1]
; CHECK-NEXT:    vmov.u16 r2, d16[2]
; CHECK-NEXT:    vmov.u16 r3, d16[3]
; CHECK-NEXT:    bx lr
	%tmp3 = call <4 x half> @llvm.roundeven.v4f16(<4 x half> %A)
	ret <4 x half> %tmp3
}

define <2 x float> @roundeven_2s(<2 x float> %A) nounwind {
; CHECK-LABEL: roundeven_2s:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d0, r0, r1
; CHECK-NEXT:    vrintn.f32 s3, s1
; CHECK-NEXT:    vrintn.f32 s2, s0
; CHECK-NEXT:    vmov r0, r1, d1
; CHECK-NEXT:    bx lr
	%tmp3 = call <2 x float> @llvm.roundeven.v2f32(<2 x float> %A)
	ret <2 x float> %tmp3
}

define <4 x float> @roundeven_4s(<4 x float> %A) nounwind {
; CHECK-LABEL: roundeven_4s:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d1, r2, r3
; CHECK-NEXT:    vmov d0, r0, r1
; CHECK-NEXT:    vrintn.f32 s7, s3
; CHECK-NEXT:    vrintn.f32 s6, s2
; CHECK-NEXT:    vrintn.f32 s5, s1
; CHECK-NEXT:    vrintn.f32 s4, s0
; CHECK-NEXT:    vmov r2, r3, d3
; CHECK-NEXT:    vmov r0, r1, d2
; CHECK-NEXT:    bx lr
	%tmp3 = call <4 x float> @llvm.roundeven.v4f32(<4 x float> %A)
	ret <4 x float> %tmp3
}

define <2 x double> @roundeven_2d(<2 x double> %A) nounwind {
; CHECK-LABEL: roundeven_2d:
; CHECK:       @ %bb.0:
; CHECK-NEXT:    vmov d16, r2, r3
; CHECK-NEXT:    vmov d17, r0, r1
; CHECK-NEXT:    vrintn.f64 d16, d16
; CHECK-NEXT:    vrintn.f64 d17, d17
; CHECK-NEXT:    vmov r2, r3, d16
; CHECK-NEXT:    vmov r0, r1, d17
; CHECK-NEXT:    bx lr
	%tmp3 = call <2 x double> @llvm.roundeven.v2f64(<2 x double> %A)
	ret <2 x double> %tmp3
}

declare <4 x half> @llvm.roundeven.v4f16(<4 x half>) nounwind readnone
declare <2 x float> @llvm.roundeven.v2f32(<2 x float>) nounwind readnone
declare <4 x float> @llvm.roundeven.v4f32(<4 x float>) nounwind readnone
declare <2 x double> @llvm.roundeven.v2f64(<2 x double>) nounwind readnone
