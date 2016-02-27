; RUN: llc -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Following test case checks:
;   a / D; b / D; c / D;
;                =>
;   recip = 1.0 / D; a * recip; b * recip; c * recip;

define void @three_fdiv_double(double %D, double %a, double %b, double %c) #0 {
; CHECK-LABEL: three_fdiv_double:
; CHECK: fdiv {{[0-9]}}
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

define void @two_fdiv_double(double %D, double %a, double %b) #0 {
; CHECK-LABEL: two_fdiv_double:
; CHECK: fdiv {{[0-9]}}
; CHECK: fdiv {{[0-9]}}
; CHECK-NOT: fmul
  %div = fdiv double %a, %D
  %div1 = fdiv double %b, %D
  tail call void @foo_2d(double %div, double %div1)
  ret void
}

declare void @foo_3d(double, double, double)
declare void @foo_3_2xd(<2 x double>, <2 x double>, <2 x double>)
declare void @foo_2d(double, double)

attributes #0 = { "unsafe-fp-math"="true" }
