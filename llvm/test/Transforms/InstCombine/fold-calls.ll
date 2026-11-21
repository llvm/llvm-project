; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This shouldn't fold, because sin(inf) is invalid.
; CHECK-LABEL: @foo(
; CHECK:   %t = call double @sin(double +inf)
define double @foo() {
  %t = call double @sin(double +inf)
  ret double %t
}

; This should fold.
; CHECK-LABEL: @bar(
; CHECK:   ret double 0.000000e+00
define double @bar() {
  %t = call double @sin(double 0.0)
  ret double %t
}

; The nearest double result is one ulp above what some host libms return.
define double @sin_hard_to_round() {
; CHECK-LABEL: @sin_hard_to_round(
; CHECK: ret double f0x3FED109AD145C88F
  %r = call double @sin(double f0x400005023D32FEE5)
  ret double %r
}

declare double @sin(double)
