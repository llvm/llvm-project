; RUN: opt < %s -instcombine -S | FileCheck %s

declare double @acos(double)

; Check that functions without any function attributes are simplified.

define double @test_simplify_acos() {
; CHECK-LABEL: @test_simplify_acos
  %pi = call double @acos(double -1.000000e+00)
; CHECK-NOT: call double @acos
; CHECK: ret double 0x400921FB54442D18
  ret double %pi
}

define double @test_acos_nobuiltin() {
; CHECK-LABEL: @test_acos_nobuiltin
  %pi = call double @acos(double -1.000000e+00) nobuiltin 
; CHECK: call double @acos(double -1.000000e+00)
  ret double %pi
}
