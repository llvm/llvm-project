; RUN: opt -S -passes=normalize < %s | FileCheck %s

define double @foo(double %a0, double %a1) {
entry:
; CHECK: %b
; CHECK: %d
; CHECK: %a
; CHECK: %c
  %a = fmul double %a0, %a1
  %b = fmul double %a0, 2.000000e+00
  %c = fmul double %a, 6.000000e+00
  %d = fmul double %b, 6.000000e+00
  ret double %d
}
