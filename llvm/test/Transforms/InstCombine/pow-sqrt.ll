; RUN: opt < %s -instcombine -S | FileCheck %s

define double @mypow(double %x) #0 {
entry:
  %pow = call double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

; CHECK-LABEL: define double @mypow(
; CHECK:   %sqrt = call double @sqrt(double %x) #1
; CHECK:   ret double %sqrt
; CHECK: }

declare double @llvm.pow.f64(double, double)
attributes #0 = { "unsafe-fp-math"="true" }
