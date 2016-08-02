; RUN: llc -mtriple=thumbv7k-apple-watchos2.0 -o - %s | FileCheck %s

declare double @sin(double) nounwind readnone
declare double @cos(double) nounwind readnone

define double @test_stret(double %in) {
; CHECK-LABEL: test_stret:
; CHECK: blx ___sincos_stret
; CHECK-NOT: ldr
; CHECK: vadd.f64 d0, d0, d1

  %sin = call double @sin(double %in)
  %cos = call double @cos(double %in)
  %sum = fadd double %sin, %cos
  ret double %sum
}
