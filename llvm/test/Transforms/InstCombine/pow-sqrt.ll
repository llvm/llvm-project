; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_half(double %x) {
  %pow = call fast double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

; CHECK-LABEL: define double @pow_half(
; CHECK-NEXT:  %sqrt = call fast double @sqrt(double %x)
; CHECK-NEXT:  ret double %sqrt

declare double @llvm.pow.f64(double, double)

