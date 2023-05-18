; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -o - %s | FileCheck %s

define float @test_sincos_f32(float %f) {
; CHECK-LABEL: test_sincos_f32:
  %sin = call float @sinf(float %f) readnone
  %cos = call float @cosf(float %f) readnone
; CHECK: bl sincosf
  %val = fadd float %sin, %cos
  ret float %val
}

define float @test_sincos_f32_errno(float %f) {
; CHECK-LABEL: test_sincos_f32_errno:
  %sin = call float @sinf(float %f)
  %cos = call float @cosf(float %f)
; CHECK: bl sinf
; CHECK: bl cosf
  %val = fadd float %sin, %cos
  ret float %val
}

define double @test_sincos_f64(double %f) {
; CHECK-LABEL: test_sincos_f64:
  %sin = call double @sin(double %f) readnone
  %cos = call double @cos(double %f) readnone
  %val = fadd double %sin, %cos
; CHECK: bl sincos
  ret double %val
}

define double @test_sincos_f64_errno(double %f) {
; CHECK-LABEL: test_sincos_f64_errno:
  %sin = call double @sin(double %f)
  %cos = call double @cos(double %f)
  %val = fadd double %sin, %cos
; CHECK: bl sin
; CHECK: bl cos
  ret double %val
}

define fp128 @test_sincos_f128(fp128 %f) {
; CHECK-LABEL: test_sincos_f128:
  %sin = call fp128 @sinl(fp128 %f) readnone
  %cos = call fp128 @cosl(fp128 %f) readnone
  %val = fadd fp128 %sin, %cos
; CHECK: bl sincosl
  ret fp128 %val
}

define fp128 @test_sincos_f128_errno(fp128 %f) {
; CHECK-LABEL: test_sincos_f128_errno:
  %sin = call fp128 @sinl(fp128 %f)
  %cos = call fp128 @cosl(fp128 %f)
  %val = fadd fp128 %sin, %cos
; CHECK: bl sinl
; CHECK: bl cosl
  ret fp128 %val
}

declare float  @sinf(float)
declare double @sin(double)
declare fp128 @sinl(fp128)
declare float @cosf(float)
declare double @cos(double)
declare fp128 @cosl(fp128)
