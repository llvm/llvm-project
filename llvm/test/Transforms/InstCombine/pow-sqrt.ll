; RUN: opt < %s -instcombine -S | FileCheck %s

define double @pow_half(double %x) {
  %pow = call fast double @llvm.pow.f64(double %x, double 5.000000e-01)
  ret double %pow
}

; CHECK-LABEL: define double @pow_half(
; CHECK-NEXT:  %sqrt = call fast double @sqrt(double %x) #1
; CHECK-NEXT:  ret double %sqrt

define double @pow_neghalf(double %x) {
  %pow = call fast double @llvm.pow.f64(double %x, double -5.000000e-01)
  ret double %pow
}

; CHECK-LABEL: define double @pow_neghalf(
; CHECK-NEXT: %sqrt = call fast double @sqrt(double %x) #1
; CHECK-NEXT: %sqrtrecip = fdiv fast double 1.000000e+00, %sqrt
; CHECK-NEXT: ret double %sqrtrecip

declare double @llvm.pow.f64(double, double) #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind readnone }
