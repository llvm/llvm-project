; RUN: opt < %s -instcombine -S | FileCheck %s

define double @std_cabs([2 x double] %z) {
; CHECK-LABEL: define double @std_cabs(
; CHECK: tail call double @cabs(
  %call = tail call double @cabs([2 x double] %z)
  ret double %call
}

define float @std_cabsf([2 x float] %z) {
; CHECK-LABEL: define float @std_cabsf(
; CHECK: tail call float @cabsf(
  %call = tail call float @cabsf([2 x float] %z)
  ret float %call
}

define fp128 @std_cabsl([2 x fp128] %z) {
; CHECK-LABEL: define fp128 @std_cabsl(
; CHECK: tail call fp128 @cabsl(
  %call = tail call fp128 @cabsl([2 x fp128] %z)
  ret fp128 %call
}

define double @fast_cabs([2 x double] %z) {
; CHECK-LABEL: define double @fast_cabs(
; CHECK: %real = extractvalue [2 x double] %z, 0
; CHECK: %imag = extractvalue [2 x double] %z, 1
; CHECK: %1 = fmul fast double %real, %real
; CHECK: %2 = fmul fast double %imag, %imag
; CHECK: %3 = fadd fast double %1, %2
; CHECK: %cabs = call fast double @llvm.sqrt.f64(double %3)
; CHECK: ret double %cabs
  %call = tail call fast double @cabs([2 x double] %z)
  ret double %call
}

define float @fast_cabsf([2 x float] %z) {
; CHECK-LABEL: define float @fast_cabsf(
; CHECK: %real = extractvalue [2 x float] %z, 0
; CHECK: %imag = extractvalue [2 x float] %z, 1
; CHECK: %1 = fmul fast float %real, %real
; CHECK: %2 = fmul fast float %imag, %imag
; CHECK: %3 = fadd fast float %1, %2
; CHECK: %cabs = call fast float @llvm.sqrt.f32(float %3)
; CHECK: ret float %cabs
  %call = tail call fast float @cabsf([2 x float] %z)
  ret float %call
}

define fp128 @fast_cabsl([2 x fp128] %z) {
; CHECK-LABEL: define fp128 @fast_cabsl(
; CHECK: %real = extractvalue [2 x fp128] %z, 0
; CHECK: %imag = extractvalue [2 x fp128] %z, 1
; CHECK: %1 = fmul fast fp128 %real, %real
; CHECK: %2 = fmul fast fp128 %imag, %imag
; CHECK: %3 = fadd fast fp128 %1, %2
; CHECK: %cabs = call fast fp128 @llvm.sqrt.f128(fp128 %3)
; CHECK: ret fp128 %cabs
  %call = tail call fast fp128 @cabsl([2 x fp128] %z)
  ret fp128 %call
}

declare double @cabs([2 x double])
declare float @cabsf([2 x float])
declare fp128 @cabsl([2 x fp128])
