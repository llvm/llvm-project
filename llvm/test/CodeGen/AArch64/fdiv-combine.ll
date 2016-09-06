; RUN: llc -mtriple=aarch64-unknown-unknown < %s | FileCheck %s

; Following test cases check:
;   a / D; b / D; c / D;
;                =>
;   recip = 1.0 / D; a * recip; b * recip; c * recip;
define void @three_fdiv_float(float %D, float %a, float %b, float %c) #0 {
; CHECK-LABEL: three_fdiv_float:
; CHECK: fdiv s
; CHECK-NOT: fdiv
; CHECK: fmul
; CHECK: fmul
; CHECK: fmul
  %div = fdiv float %a, %D
  %div1 = fdiv float %b, %D
  %div2 = fdiv float %c, %D
  tail call void @foo_3f(float %div, float %div1, float %div2)
  ret void
}

define void @three_fdiv_double(double %D, double %a, double %b, double %c) #0 {
; CHECK-LABEL: three_fdiv_double:
; CHECK: fdiv d
; CHECK-NOT: fdiv
; CHECK: fmul
; CHECK: fmul
; CHECK: fmul
  %div = fdiv double %a, %D
  %div1 = fdiv double %b, %D
  %div2 = fdiv double %c, %D
  tail call void @foo_3d(double %div, double %div1, double %div2)
  ret void
}

define void @three_fdiv_4xfloat(<4 x float> %D, <4 x float> %a, <4 x float> %b, <4 x float> %c) #0 {
; CHECK-LABEL: three_fdiv_4xfloat:
; CHECK: fdiv v
; CHECK-NOT: fdiv
; CHECK: fmul
; CHECK: fmul
; CHECK: fmul
  %div = fdiv <4 x float> %a, %D
  %div1 = fdiv <4 x float> %b, %D
  %div2 = fdiv <4 x float> %c, %D
  tail call void @foo_3_4xf(<4 x float> %div, <4 x float> %div1, <4 x float> %div2)
  ret void
}

define void @three_fdiv_2xdouble(<2 x double> %D, <2 x double> %a, <2 x double> %b, <2 x double> %c) #0 {
; CHECK-LABEL: three_fdiv_2xdouble:
; CHECK: fdiv v
; CHECK-NOT: fdiv
; CHECK: fmul
; CHECK: fmul
; CHECK: fmul
  %div = fdiv <2 x double> %a, %D
  %div1 = fdiv <2 x double> %b, %D
  %div2 = fdiv <2 x double> %c, %D
  tail call void @foo_3_2xd(<2 x double> %div, <2 x double> %div1, <2 x double> %div2)
  ret void
}

; Following test cases check we never combine two FDIVs if neither of them
; calculates a reciprocal.
define void @two_fdiv_float(float %D, float %a, float %b) #0 {
; CHECK-LABEL: two_fdiv_float:
; CHECK: fdiv s
; CHECK: fdiv s
; CHECK-NOT: fmul
  %div = fdiv float %a, %D
  %div1 = fdiv float %b, %D
  tail call void @foo_2f(float %div, float %div1)
  ret void
}

define void @two_fdiv_double(double %D, double %a, double %b) #0 {
; CHECK-LABEL: two_fdiv_double:
; CHECK: fdiv d
; CHECK: fdiv d
; CHECK-NOT: fmul
  %div = fdiv double %a, %D
  %div1 = fdiv double %b, %D
  tail call void @foo_2d(double %div, double %div1)
  ret void
}

declare void @foo_3f(float, float, float)
declare void @foo_3d(double, double, double)
declare void @foo_3_4xf(<4 x float>, <4 x float>, <4 x float>)
declare void @foo_3_2xd(<2 x double>, <2 x double>, <2 x double>)
declare void @foo_2f(float, float)
declare void @foo_2d(double, double)

attributes #0 = { "unsafe-fp-math"="true" }
