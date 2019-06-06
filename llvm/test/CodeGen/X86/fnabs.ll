; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=avx| FileCheck %s

; Verify that we generate a single OR instruction for a scalar, vec128, and vec256
; FNABS(x) operation -> FNEG (FABS(x)).
; If the FABS() result isn't used, the AND instruction should be eliminated.
; PR20578: http://llvm.org/bugs/show_bug.cgi?id=20578

define float @scalar_no_abs(float %a) {
; CHECK-LABEL: scalar_no_abs:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call float @fabsf(float %a) #1
  %fsub = fsub float -0.0, %fabs
  ret float %fsub
}

define float @scalar_no_abs_unary_fneg(float %a) {
; CHECK-LABEL: scalar_no_abs_unary_fneg:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call float @fabsf(float %a) #1
  %fsub = fneg float %fabs
  ret float %fsub
}

define float @scalar_uses_abs(float %a) {
; CHECK-LABEL: scalar_uses_abs:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulss
; CHECK-NEXT: retq
  %fabs = tail call float @fabsf(float %a) #1
  %fsub = fsub float -0.0, %fabs
  %fmul = fmul float %fsub, %fabs
  ret float %fmul
}

define float @scalar_uses_abs_unary_fneg(float %a) {
; CHECK-LABEL: scalar_uses_abs_unary_fneg:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulss
; CHECK-NEXT: retq
  %fabs = tail call float @fabsf(float %a) #1
  %fsub = fneg float %fabs
  %fmul = fmul float %fsub, %fabs
  ret float %fmul
}

define <4 x float> @vector128_no_abs(<4 x float> %a) {
; CHECK-LABEL: vector128_no_abs:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call <4 x float> @llvm.fabs.v4f32(< 4 x float> %a) #1
  %fsub = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %fabs
  ret <4 x float> %fsub
}

define <4 x float> @vector128_no_abs_unary_fneg(<4 x float> %a) {
; CHECK-LABEL: vector128_no_abs_unary_fneg:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call <4 x float> @llvm.fabs.v4f32(< 4 x float> %a) #1
  %fsub = fneg <4 x float> %fabs
  ret <4 x float> %fsub
}

define <4 x float> @vector128_uses_abs(<4 x float> %a) {
; CHECK-LABEL: vector128_uses_abs:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulps
; CHECK-NEXT: retq
  %fabs = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %a) #1
  %fsub = fsub <4 x float> <float -0.0, float -0.0, float -0.0, float -0.0>, %fabs
  %fmul = fmul <4 x float> %fsub, %fabs
  ret <4 x float> %fmul
}

define <4 x float> @vector128_uses_abs_unary_fneg(<4 x float> %a) {
; CHECK-LABEL: vector128_uses_abs_unary_fneg:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulps
; CHECK-NEXT: retq
  %fabs = tail call <4 x float> @llvm.fabs.v4f32(<4 x float> %a) #1
  %fsub = fneg <4 x float> %fabs
  %fmul = fmul <4 x float> %fsub, %fabs
  ret <4 x float> %fmul
}

define <8 x float> @vector256_no_abs(<8 x float> %a) {
; CHECK-LABEL: vector256_no_abs:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call <8 x float> @llvm.fabs.v8f32(< 8 x float> %a) #1
  %fsub = fsub <8 x float> <float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0>, %fabs
  ret <8 x float> %fsub
}

define <8 x float> @vector256_no_abs_unary_fneg(<8 x float> %a) {
; CHECK-LABEL: vector256_no_abs_unary_fneg:
; CHECK: vorps
; CHECK-NEXT: retq
  %fabs = tail call <8 x float> @llvm.fabs.v8f32(< 8 x float> %a) #1
  %fsub = fneg <8 x float> %fabs
  ret <8 x float> %fsub
}

define <8 x float> @vector256_uses_abs(<8 x float> %a) {
; CHECK-LABEL: vector256_uses_abs:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulps
; CHECK-NEXT: retq
  %fabs = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %a) #1
  %fsub = fsub <8 x float> <float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0, float -0.0>, %fabs
  %fmul = fmul <8 x float> %fsub, %fabs
  ret <8 x float> %fmul
}

define <8 x float> @vector256_uses_abs_unary_fneg(<8 x float> %a) {
; CHECK-LABEL: vector256_uses_abs_unary_fneg:
; CHECK-DAG: vandps
; CHECK-DAG: vorps
; CHECK: vmulps
; CHECK-NEXT: retq
  %fabs = tail call <8 x float> @llvm.fabs.v8f32(<8 x float> %a) #1
  %fsub = fneg <8 x float> %fabs
  %fmul = fmul <8 x float> %fsub, %fabs
  ret <8 x float> %fmul
}

declare <4 x float> @llvm.fabs.v4f32(<4 x float> %p)
declare <8 x float> @llvm.fabs.v8f32(<8 x float> %p)

declare float @fabsf(float)

attributes #1 = { readnone }

