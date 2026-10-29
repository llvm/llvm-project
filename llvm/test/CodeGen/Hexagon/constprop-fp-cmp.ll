; RUN: llc -mtriple=hexagon -O2 < %s | FileCheck %s

; Test coverage for HexagonConstPropagation: exercise floating-point
; constant propagation paths by using FP comparisons and operations
; with known constants.

; CHECK-LABEL: test_fp_const_fold:
; CHECK: jumpr r31
define i32 @test_fp_const_fold() {
entry:
  %cmp = fcmp ogt float 2.0, 1.0
  %val = select i1 %cmp, i32 42, i32 0
  ret i32 %val
}

; CHECK-LABEL: test_fp_zero_cmp:
; CHECK: jumpr r31
define i32 @test_fp_zero_cmp() {
entry:
  %cmp = fcmp oeq float 0.0, 0.0
  %val = select i1 %cmp, i32 1, i32 0
  ret i32 %val
}

; CHECK-LABEL: test_fp_nan_cmp:
; CHECK: jumpr r31
define i32 @test_fp_nan_cmp() {
entry:
  %nan = fdiv float 0.0, 0.0
  %cmp = fcmp uno float %nan, 1.0
  %val = select i1 %cmp, i32 1, i32 0
  ret i32 %val
}

; Exercise the negative FP constant path.
; CHECK-LABEL: test_fp_neg:
; CHECK: jumpr r31
define i32 @test_fp_neg(float %x) {
entry:
  %neg = fneg float %x
  %cmp = fcmp olt float %neg, 0.0
  %val = select i1 %cmp, i32 1, i32 0
  ret i32 %val
}

