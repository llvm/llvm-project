; RUN: opt < %s -mtriple=amdgcn-- -targetlibinfo -instcombine -S | FileCheck %s

; Check that a transform of llvm.pow to multiplies or divides happens for amdgcn (for which
; TLI says no lib functions).

define float @pow_of_1(float %x) {
; CHECK-LABEL: @pow_of_1(
; CHECK-NEXT: ret float
;
  %pow = call float @llvm.pow.f32(float 1.0, float %x)
  ret float %pow
}

define float @pow_minus_1(float %x) {
; CHECK-LABEL: @pow_minus_1(
; CHECK: = fdiv float
;
  %pow = call float @llvm.pow.f32(float %x, float -1.0)
  ret float %pow
}

define float @pow_0(float %x) {
; CHECK-LABEL: @pow_0(
; CHECK-NEXT: ret float
;
  %pow = call float @llvm.pow.f32(float %x, float 0.0)
  ret float %pow
}

define float @pow_1(float %x) {
; CHECK-LABEL: @pow_1(
; CHECK-NEXT: ret float
;
  %pow = call float @llvm.pow.f32(float %x, float 1.0)
  ret float %pow
}

define float @pow_half(float %x) {
; CHECK-LABEL: @pow_half(
; CHECK: @llvm.sqrt
;
  %pow = call float @llvm.pow.f32(float %x, float 0.5)
  ret float %pow
}

define float @pow_2(float %x) {
; CHECK-LABEL: @pow_2(
; CHECK: = fmul float
;
  %pow = call float @llvm.pow.f32(float %x, float 2.0)
  ret float %pow
}

declare float @llvm.pow.f32(float, float)

