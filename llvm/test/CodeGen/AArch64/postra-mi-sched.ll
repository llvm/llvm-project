; RUN: llc < %s -O3 -mtriple=aarch64-eabi -mcpu=cortex-a53 | FileCheck %s

; With cortex-a53, each of fmul and fcvt have latency of 6 cycles.  After the
; pre-RA MI scheduler, fmul, fcvt and fdiv will be consecutive.  The top-down
; post-RA MI scheduler will clean this up.

@d1 = common global double 0.000000e+00, align 8

define i32 @test1(float %s2, float %s3, double %d, i32 %i2, i32 %i3) {
entry:
; CHECK-LABEL: @test1
; CHECK: fmul
; CHECK-NEXT: add
; CHECK: fcvt
; CHECK-NEXT: mul
  %mul = fmul float %s2, %s3
  %conv = fpext float %mul to double
  %div = fdiv double %d, %conv
  store double %div, double* @d1, align 8
  %factor = shl i32 %i3, 1
  %add1 = add i32 %i2, 4
  %add2 = add i32 %add1, %factor
  %add3 = add nsw i32 %add2, %i2
  %add4 = add nsw i32 %add3, %add2
  %mul5 = mul i32 %add3, %add3
  %mul6 = mul i32 %mul5, %add4
  %mul7 = shl i32 %add4, 1
  %factor18 = mul i32 %mul7, %mul6
  %add9 = add i32 %factor18, %mul6
  ret i32 %add9
}
