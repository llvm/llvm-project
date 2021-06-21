; RUN: llc -mtriple x86_64-windows < %s | FileCheck %s

declare double @llvm.powi.f64.i32(double, i32)
declare float @llvm.powi.f32.i32(float, i32)

define double @d(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64.i32(double %d, i32 %i)
  ret double %0
}

; CHECK-LABEL: d:
; CHECK: cvtsi2sd %edx, %xmm1
; CHECK-NEXT: jmp pow

define float @f(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32.i32(float %f, i32 %i)
  ret float %0
}

; CHECK-LABEL: f:
; CHECK: cvtsi2ss %edx, %xmm1
; CHECK-NEXT: jmp powf

define float @g(double %d, i32 %i) {
entry:
  %0 = tail call double @llvm.powi.f64.i32(double %d, i32 %i)
  %conv = fptrunc double %0 to float
  ret float %conv
}

; CHECK-LABEL: g:
; CHECK: cvtsi2sd %edx, %xmm1
; CHECK-NEXT: callq pow

define double @h(float %f, i32 %i) {
entry:
  %0 = tail call float @llvm.powi.f32.i32(float %f, i32 %i)
  %conv = fpext float %0 to double
  ret double %conv
}

; CHECK-LABEL: h:
; CHECK: cvtsi2ss %edx, %xmm1
; CHECK-NEXT: callq powf
