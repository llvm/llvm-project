; RUN: opt -instcombine -S < %s | FileCheck %s

define double @test1() {
  %sin = call double @__sinpi(double 1.0)
  ret double %sin
}

; CHECK-LABEL: define double @test1(
; CHECK: %[[sin:.*]] = call double @__sinpi(double 1.000000e+00)
; CHECK-NEXT: ret double %[[sin]]

define double @test2() {
  %cos = call double @__cospi(double 1.0)
  ret double %cos
}

; CHECK-LABEL: define double @test2(
; CHECK: %[[cos:.*]] = call double @__cospi(double 1.000000e+00)
; CHECK-NEXT: ret double %[[cos]]

declare double @__sinpi(double %x) #0
declare double @__cospi(double %x) #0

attributes #0 = { readnone nounwind }
