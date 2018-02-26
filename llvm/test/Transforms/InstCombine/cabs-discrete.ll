; RUN: opt < %s -instcombine -S | FileCheck %s

define double @std_cabs(double %real, double %imag) {
; CHECK-LABEL: define double @std_cabs(
; CHECK: tail call double @cabs(
  %call = tail call double @cabs(double %real, double %imag)
  ret double %call
}

define float @std_cabsf(float %real, float %imag) {
; CHECK-LABEL: define float @std_cabsf(
; CHECK: tail call float @cabsf(
  %call = tail call float @cabsf(float %real, float %imag)
  ret float %call
}

define fp128 @std_cabsl(fp128 %real, fp128 %imag) {
; CHECK-LABEL: define fp128 @std_cabsl(
; CHECK: tail call fp128 @cabsl(
  %call = tail call fp128 @cabsl(fp128 %real, fp128 %imag)
  ret fp128 %call
}

define double @fast_cabs(double %real, double %imag) {
; CHECK-LABEL: define double @fast_cabs(
; CHECK: %1 = fmul fast double %real, %real
; CHECK: %2 = fmul fast double %imag, %imag
; CHECK: %3 = fadd fast double %1, %2
; CHECK: %cabs = call fast double @llvm.sqrt.f64(double %3)
; CHECK: ret double %cabs
  %call = tail call fast double @cabs(double %real, double %imag)
  ret double %call
}

define float @fast_cabsf(float %real, float %imag) {
; CHECK-LABEL: define float @fast_cabsf(
; CHECK: %1 = fmul fast float %real, %real
; CHECK: %2 = fmul fast float %imag, %imag
; CHECK: %3 = fadd fast float %1, %2
; CHECK: %cabs = call fast float @llvm.sqrt.f32(float %3)
; CHECK: ret float %cabs
  %call = tail call fast float @cabsf(float %real, float %imag)
  ret float %call
}

define fp128 @fast_cabsl(fp128 %real, fp128 %imag) {
; CHECK-LABEL: define fp128 @fast_cabsl(
; CHECK: %1 = fmul fast fp128 %real, %real
; CHECK: %2 = fmul fast fp128 %imag, %imag
; CHECK: %3 = fadd fast fp128 %1, %2
; CHECK: %cabs = call fast fp128 @llvm.sqrt.f128(fp128 %3)
; CHECK: ret fp128 %cabs
  %call = tail call fast fp128 @cabsl(fp128 %real, fp128 %imag)
  ret fp128 %call
}

declare double @cabs(double %real, double %imag)
declare float @cabsf(float %real, float %imag)
declare fp128 @cabsl(fp128 %real, fp128 %imag)
