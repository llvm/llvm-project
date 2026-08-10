; RUN: llc -mtriple=arm64-apple-darwin                             -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=arm64-apple-darwin -fast-isel -fast-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s

define float @test_sqrt_f32(float %a) {
; CHECK-LABEL: test_sqrt_f32
; CHECK:       fsqrt s0, s0
  %res = call float @llvm.sqrt.f32(float %a)
  ret float %res
}
declare float @llvm.sqrt.f32(float) nounwind readnone

define double @test_sqrt_f64(double %a) {
; CHECK-LABEL: test_sqrt_f64
; CHECK:       fsqrt d0, d0
  %res = call double @llvm.sqrt.f64(double %a)
  ret double %res
}
declare double @llvm.sqrt.f64(double) nounwind readnone


