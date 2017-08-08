; Test that the pow library call simplifier works correctly.

; RUN: opt -instcombine -S < %s | FileCheck %s

; Function Attrs: nounwind readnone
declare double @llvm.pow.f64(double, double)
declare float @llvm.pow.f32(float, float)

; pow(x, 4.0f)
define float @test_simplify_4f(float %x) {
; CHECK-LABEL: @test_simplify_4f(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast float %x, %x
; CHECK-NEXT: %2 = fmul fast float %1, %1
; CHECK-NEXT: ret float %2
  %1 = call fast float @llvm.pow.f32(float %x, float 4.000000e+00)
  ret float %1
}

; pow(x, 3.0)
define double @test_simplify_3(double %x) {
; CHECK-LABEL: @test_simplify_3(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %x
; CHECK-NEXT: ret double %2
  %1 = call fast double @llvm.pow.f64(double %x, double 3.000000e+00)
  ret double %1
}

; pow(x, 4.0)
define double @test_simplify_4(double %x) {
; CHECK-LABEL: @test_simplify_4(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %1
; CHECK-NEXT: ret double %2
  %1 = call fast double @llvm.pow.f64(double %x, double 4.000000e+00)
  ret double %1
}

; pow(x, 15.0)
define double @test_simplify_15(double %x) {
; CHECK-LABEL: @test_simplify_15(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %x
; CHECK-NEXT: %3 = fmul fast double %2, %2
; CHECK-NEXT: %4 = fmul fast double %3, %3
; CHECK-NEXT: %5 = fmul fast double %2, %4
; CHECK-NEXT: ret double %5
  %1 = call fast double @llvm.pow.f64(double %x, double 1.500000e+01)
  ret double %1
}

; pow(x, -7.0)
define double @test_simplify_neg_7(double %x) {
; CHECK-LABEL: @test_simplify_neg_7(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %1
; CHECK-NEXT: %3 = fmul fast double %2, %x
; CHECK-NEXT: %4 = fmul fast double %1, %3
; CHECK-NEXT: %5 = fdiv fast double 1.000000e+00, %4
; CHECK-NEXT: ret double %5
  %1 = call fast double @llvm.pow.f64(double %x, double -7.000000e+00)
  ret double %1
}

; pow(x, -19.0)
define double @test_simplify_neg_19(double %x) {
; CHECK-LABEL: @test_simplify_neg_19(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %1
; CHECK-NEXT: %3 = fmul fast double %2, %2
; CHECK-NEXT: %4 = fmul fast double %3, %3
; CHECK-NEXT: %5 = fmul fast double %1, %4
; CHECK-NEXT: %6 = fmul fast double %5, %x
; CHECK-NEXT: %7 = fdiv fast double 1.000000e+00, %6
; CHECK-NEXT: ret double %7
  %1 = call fast double @llvm.pow.f64(double %x, double -1.900000e+01)
  ret double %1
}

; pow(x, 11.23)
define double @test_simplify_11_23(double %x) {
; CHECK-LABEL: @test_simplify_11_23(
; CHECK-NOT: fmul
; CHECK-NEXT: %1 = call fast double @llvm.pow.f64(double %x, double 1.123000e+01)
; CHECK-NEXT: ret double %1
  %1 = call fast double @llvm.pow.f64(double %x, double 1.123000e+01)
  ret double %1
}

; pow(x, 32.0)
define double @test_simplify_32(double %x) {
; CHECK-LABEL: @test_simplify_32(
; CHECK-NOT: pow
; CHECK-NEXT: %1 = fmul fast double %x, %x
; CHECK-NEXT: %2 = fmul fast double %1, %1
; CHECK-NEXT: %3 = fmul fast double %2, %2
; CHECK-NEXT: %4 = fmul fast double %3, %3
; CHECK-NEXT: %5 = fmul fast double %4, %4
; CHECK-NEXT: ret double %5
  %1 = call fast double @llvm.pow.f64(double %x, double 3.200000e+01)
  ret double %1
}

; pow(x, 33.0)
define double @test_simplify_33(double %x) {
; CHECK-LABEL: @test_simplify_33(
; CHECK-NOT: fmul
; CHECK-NEXT: %1 = call fast double @llvm.pow.f64(double %x, double 3.300000e+01)
; CHECK-NEXT: ret double %1
  %1 = call fast double @llvm.pow.f64(double %x, double 3.300000e+01)
  ret double %1
}

