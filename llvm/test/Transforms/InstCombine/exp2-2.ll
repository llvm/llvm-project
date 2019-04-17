; Test that the exp2 library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

declare float @exp2(double)

; Check that exp2 functions with the wrong prototype aren't simplified.

define float @test_no_simplify1(i32 %x) {
; CHECK-LABEL: @test_no_simplify1(
  %conv = sitofp i32 %x to double
  %ret = call float @exp2(double %conv)
; CHECK: call float @exp2(double %conv)
  ret float %ret
}
