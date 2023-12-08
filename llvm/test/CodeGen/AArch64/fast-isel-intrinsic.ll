; RUN: llc -mtriple=aarch64-apple-darwin            -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs < %s | FileCheck %s

define float @fabs_f32(float %a) {
; CHECK-LABEL: fabs_f32
; CHECK:       fabs s0, s0
  %1 = call float @llvm.fabs.f32(float %a)
  ret float %1
}

define double @fabs_f64(double %a) {
; CHECK-LABEL: fabs_f64
; CHECK:       fabs d0, d0
  %1 = call double @llvm.fabs.f64(double %a)
  ret double %1
}

declare double @llvm.fabs.f64(double)
declare float @llvm.fabs.f32(float)
