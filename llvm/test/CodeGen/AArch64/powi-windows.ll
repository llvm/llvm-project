; RUN: llc -mtriple aarch64-windows < %s | FileCheck %s

declare double @llvm.powi.f64(double, i32)
declare float @llvm.powi.f32(float, i32)

define double @d(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64(double %d, i32 %i)
  ret double %0
}

; CHECK-LABEL: d:
; CHECK: scvtf d1, w0
; CHECK-NEXT: b pow

define float @f(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32(float %f, i32 %i)
  ret float %0
}

; CHECK-LABEL: f:
; CHECK: scvtf s1, w0
; CHECK-NEXT: b powf

define float @g(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64(double %d, i32 %i)
  %conv = fptrunc double %0 to float
  ret float %conv
}

; CHECK-LABEL: g:
; CHECK: scvtf d1, w0
; CHECK-NEXT: bl pow

define double @h(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32(float %f, i32 %i)
  %conv = fpext float %0 to double
  ret double %conv
}

; CHECK-LABEL: h:
; CHECK: scvtf s1, w0
; CHECK-NEXT: bl powf
