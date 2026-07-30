; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare float @cos(double)
declare signext i8 @sqrt(...)

; Check that functions with the wrong prototype aren't simplified.

define float @test_no_simplify1(double %d) {
; CHECK-LABEL: @test_no_simplify1(
  %neg = fsub double -0.000000e+00, %d
  %cos = call float @cos(double %neg)
; CHECK: call float @cos(double %neg)
  ret float %cos
}

define float @test_no_simplify2(double %d) {
; CHECK-LABEL: @test_no_simplify2(
  %neg = fneg double %d
  %cos = call float @cos(double %neg)
; CHECK: call float @cos(double %neg)
  ret float %cos
}

define i8 @bogus_sqrt() {
  %fake_sqrt = call signext i8 (...) @sqrt()
  ret i8 %fake_sqrt

; CHECK-LABEL: bogus_sqrt(
; CHECK-NEXT:  %fake_sqrt = call signext i8 (...) @sqrt()
; CHECK-NEXT:  ret i8 %fake_sqrt
}

