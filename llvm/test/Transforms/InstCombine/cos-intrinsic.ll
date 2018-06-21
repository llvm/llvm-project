; RUN: opt < %s -instcombine -S | FileCheck %s
; This test makes sure that the undef is propagated for the cos instrinsic

declare double    @llvm.cos.f64(double %Val)
declare float     @llvm.cos.f32(float %Val)
declare <2 x float> @llvm.cos.v2f32(<2 x float> %Val)

declare float @llvm.fabs.f32(float %Val)
declare <2 x float> @llvm.fabs.v2f32(<2 x float> %Val)

; Function Attrs: nounwind readnone
define double @test1() {
; CHECK-LABEL: define double @test1(
; CHECK-NEXT: ret double 0.000000e+00
  %1 = call double @llvm.cos.f64(double undef)
  ret double %1
}


; Function Attrs: nounwind readnone
define float @test2(float %d) {
; CHECK-LABEL: define float @test2(
; CHECK-NEXT: %cosval = call float @llvm.cos.f32(float %d)
   %cosval   = call float @llvm.cos.f32(float %d)
   %cosval2  = call float @llvm.cos.f32(float undef)
   %fsum   = fadd float %cosval2, %cosval
   ret float %fsum
; CHECK-NEXT: %fsum
; CHECK: ret float %fsum
}

; CHECK-LABEL: @cos_fneg_f32(
; CHECK: %cos = call float @llvm.cos.f32(float %x)
; CHECK-NEXT: ret float %cos
define float @cos_fneg_f32(float %x) {
  %x.fneg = fsub float -0.0, %x
  %cos = call float @llvm.cos.f32(float %x.fneg)
  ret float %cos
}

; FIXME: m_FNeg() doesn't handle vectors
; CHECK-LABEL: @cos_fneg_v2f32(
; CHECK: %x.fneg = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %x
; CHECK-NEXT: %cos = call <2 x float> @llvm.cos.v2f32(<2 x float> %x.fneg)
; CHECK-NEXT: ret <2 x float> %cos
define <2 x float> @cos_fneg_v2f32(<2 x float> %x) {
  %x.fneg = fsub <2 x float> <float -0.0, float -0.0>, %x
  %cos = call <2 x float> @llvm.cos.v2f32(<2 x float> %x.fneg)
  ret <2 x float> %cos
}

; CHECK-LABEL: @cos_fabs_f32(
; CHECK-NEXT: %cos = call float @llvm.cos.f32(float %x)
; CHECK-NEXT: ret float %cos
define float @cos_fabs_f32(float %x) {
  %x.fabs = call float @llvm.fabs.f32(float %x)
  %cos = call float @llvm.cos.f32(float %x.fabs)
  ret float %cos
}

; CHECK-LABEL: @cos_fabs_fneg_f32(
; CHECK: %cos = call float @llvm.cos.f32(float %x)
; CHECK-NEXT: ret float %cos
define float @cos_fabs_fneg_f32(float %x) {
  %x.fabs = call float @llvm.fabs.f32(float %x)
  %x.fabs.fneg = fsub float -0.0, %x.fabs
  %cos = call float @llvm.cos.f32(float %x.fabs.fneg)
  ret float %cos
}

; CHECK-LABEL: @cos_fabs_fneg_v2f32(
; CHECK: %x.fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %x)
; CHECK-NEXT: %x.fabs.fneg = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %x.fabs
; CHECK-NEXT: %cos = call <2 x float> @llvm.cos.v2f32(<2 x float> %x.fabs.fneg)
; CHECK-NEXT: ret <2 x float> %cos
define <2 x float> @cos_fabs_fneg_v2f32(<2 x float> %x) {
  %x.fabs = call <2 x float> @llvm.fabs.v2f32(<2 x float> %x)
  %x.fabs.fneg = fsub <2 x float> <float -0.0, float -0.0>, %x.fabs
  %cos = call <2 x float> @llvm.cos.v2f32(<2 x float> %x.fabs.fneg)
  ret <2 x float> %cos
}
