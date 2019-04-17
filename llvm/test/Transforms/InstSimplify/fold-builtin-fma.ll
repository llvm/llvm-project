; RUN: opt -instsimplify -S < %s | FileCheck %s

; Fixes PR20832
; Make sure that we correctly fold a fused multiply-add where operands
; are all finite constants and addend is zero.

declare double @llvm.fma.f64(double, double, double)


define double @PR20832()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 0.0)
  ret double %1
}
; CHECK-LABEL: @PR20832(
; CHECK: ret double 5.600000e+01

; Test builtin fma with all finite non-zero constants.
define double @test_all_finite()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 5.0)
  ret double %1
}
; CHECK-LABEL: @test_all_finite(
; CHECK: ret double 6.100000e+01

; Test builtin fma with a +/-NaN addend.
define double @test_NaN_addend()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 0x7FF8000000000000)
  ret double %1
}
; CHECK-LABEL: @test_NaN_addend(
; CHECK: ret double 0x7FF8000000000000

define double @test_NaN_addend_2()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 0xFFF8000000000000)
  ret double %1
}
; CHECK-LABEL: @test_NaN_addend_2(
; CHECK: ret double 0xFFF8000000000000

; Test builtin fma with a +/-Inf addend.
define double @test_Inf_addend()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 0x7FF0000000000000)
  ret double %1
}
; CHECK-LABEL: @test_Inf_addend(
; CHECK: ret double 0x7FF0000000000000

define double @test_Inf_addend_2()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 8.0, double 0xFFF0000000000000)
  ret double %1
}
; CHECK-LABEL: @test_Inf_addend_2(
; CHECK: ret double 0xFFF0000000000000

; Test builtin fma with one of the operands to the multiply being +/-NaN.
define double @test_NaN_1()  {
  %1 = call double @llvm.fma.f64(double 0x7FF8000000000000, double 8.0, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_NaN_1(
; CHECK: ret double 0x7FF8000000000000


define double @test_NaN_2()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 0x7FF8000000000000, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_NaN_2(
; CHECK: ret double 0x7FF8000000000000


define double @test_NaN_3()  {
  %1 = call double @llvm.fma.f64(double 0xFFF8000000000000, double 8.0, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_NaN_3(
; CHECK: ret double 0x7FF8000000000000


define double @test_NaN_4()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 0xFFF8000000000000, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_NaN_4(
; CHECK: ret double 0x7FF8000000000000


; Test builtin fma with one of the operands to the multiply being +/-Inf.
define double @test_Inf_1()  {
  %1 = call double @llvm.fma.f64(double 0x7FF0000000000000, double 8.0, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_Inf_1(
; CHECK: ret double 0x7FF0000000000000


define double @test_Inf_2()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 0x7FF0000000000000, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_Inf_2(
; CHECK: ret double 0x7FF0000000000000


define double @test_Inf_3()  {
  %1 = call double @llvm.fma.f64(double 0xFFF0000000000000, double 8.0, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_Inf_3(
; CHECK: ret double 0xFFF0000000000000


define double @test_Inf_4()  {
  %1 = call double @llvm.fma.f64(double 7.0, double 0xFFF0000000000000, double 0.0)
  ret double %1
}
; CHECK-LABEL: @test_Inf_4(
; CHECK: ret double 0xFFF0000000000000

