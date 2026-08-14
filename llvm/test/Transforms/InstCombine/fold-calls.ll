; RUN: opt -passes=instcombine -S < %s | FileCheck %s

; This shouldn't fold, because sin(inf) is invalid.
; CHECK-LABEL: @foo(
; CHECK:   %t = call double @sin(double +inf)
define double @foo() {
  %t = call double @sin(double 0x7FF0000000000000)
  ret double %t
}

; This should fold.
; CHECK-LABEL: @bar(
; CHECK:   ret double 0.000000e+00
define double @bar() {
  %t = call double @sin(double 0.0)
  ret double %t
}

declare double @sin(double)
