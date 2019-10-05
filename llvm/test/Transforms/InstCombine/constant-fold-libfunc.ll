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

; Check that we don't constant fold builtin functions.

define double @test_acos_nobuiltin() {
; CHECK-LABEL: @test_acos_nobuiltin
  %pi = call double @acos(double -1.000000e+00) nobuiltin 
; CHECK: call double @acos(double -1.000000e+00)
  ret double %pi
}

; Check that we don't constant fold strictfp results that require rounding.

define double @test_acos_strictfp() strictfp {
; CHECK-LABEL: @test_acos_strictfp
  %pi = call double @acos(double -1.000000e+00) strictfp 
; CHECK: call double @acos(double -1.000000e+00)
  ret double %pi
}
