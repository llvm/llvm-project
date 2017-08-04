; Test that the pow library call simplifier works correctly.
;
; RUN: opt < %s -instcombine -S | FileCheck %s
; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.9 | FileCheck %s --check-prefix=CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios7.0 | FileCheck %s --check-prefix=CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.8 | FileCheck %s --check-prefix=CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios6.0 | FileCheck %s --check-prefix=CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=x86_64-netbsd | FileCheck %s --check-prefix=CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-tvos9.0 | FileCheck %s --check-prefix=CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-watchos2.0 | FileCheck %s --check-prefix=CHECK-EXP10
; rdar://7251832

; NOTE: The readonly attribute on the pow call should be preserved
; in the cases below where pow is transformed into another function call.

declare float @powf(float, float) nounwind readonly
declare double @pow(double, double) nounwind readonly

; Check pow(1.0, x) -> 1.0.

define float @test_simplify1(float %x) {
; CHECK-LABEL: @test_simplify1(
  %retval = call float @powf(float 1.0, float %x)
  ret float %retval
; CHECK-NEXT: ret float 1.000000e+00
}

define double @test_simplify2(double %x) {
; CHECK-LABEL: @test_simplify2(
  %retval = call double @pow(double 1.0, double %x)
  ret double %retval
; CHECK-NEXT: ret double 1.000000e+00
}

; Check pow(2.0, x) -> exp2(x).

define float @test_simplify3(float %x) {
; CHECK-LABEL: @test_simplify3(
  %retval = call float @powf(float 2.0, float %x)
; CHECK-NEXT: [[EXP2F:%[a-z0-9]+]] = call float @llvm.exp2.f32(float %x)
  ret float %retval
; CHECK-NEXT: ret float [[EXP2F]]
}

define double @test_simplify4(double %x) {
; CHECK-LABEL: @test_simplify4(
  %retval = call double @pow(double 2.0, double %x)
; CHECK-NEXT: [[EXP2:%[a-z0-9]+]] = call double @llvm.exp2.f64(double %x)
  ret double %retval
; CHECK-NEXT: ret double [[EXP2]]
}

; Check pow(x, 0.0) -> 1.0.

define float @test_simplify5(float %x) {
; CHECK-LABEL: @test_simplify5(
  %retval = call float @powf(float %x, float 0.0)
  ret float %retval
; CHECK-NEXT: ret float 1.000000e+00
}

define double @test_simplify6(double %x) {
; CHECK-LABEL: @test_simplify6(
  %retval = call double @pow(double %x, double 0.0)
  ret double %retval
; CHECK-NEXT: ret double 1.000000e+00
}

; Check pow(x, 0.5) -> fabs(sqrt(x)), where x != -infinity.

define float @test_simplify7(float %x) {
; CHECK-LABEL: @test_simplify7(
  %retval = call float @powf(float %x, float 0.5)
; CHECK-NEXT: [[SQRTF:%[a-z0-9]+]] = call float @sqrtf(float %x) [[NUW_RO:#[0-9]+]]
; CHECK-NEXT: [[FABSF:%[a-z0-9]+]] = call float @llvm.fabs.f32(float [[SQRTF]])
; CHECK-NEXT: [[FCMP:%[a-z0-9]+]] = fcmp oeq float %x, 0xFFF0000000000000
; CHECK-NEXT: [[SELECT:%[a-z0-9]+]] = select i1 [[FCMP]], float 0x7FF0000000000000, float [[FABSF]]
  ret float %retval
; CHECK-NEXT: ret float [[SELECT]]
}

define double @test_simplify8(double %x) {
; CHECK-LABEL: @test_simplify8(
  %retval = call double @pow(double %x, double 0.5)
; CHECK-NEXT: [[SQRT:%[a-z0-9]+]] = call double @sqrt(double %x) [[NUW_RO]]
; CHECK-NEXT: [[FABS:%[a-z0-9]+]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT: [[FCMP:%[a-z0-9]+]] = fcmp oeq double %x, 0xFFF0000000000000
; CHECK-NEXT: [[SELECT:%[a-z0-9]+]] = select i1 [[FCMP]], double 0x7FF0000000000000, double [[FABS]]
  ret double %retval
; CHECK-NEXT: ret double [[SELECT]]
}

; Check pow(-infinity, 0.5) -> +infinity.

define float @test_simplify9(float %x) {
; CHECK-LABEL: @test_simplify9(
  %retval = call float @powf(float 0xFFF0000000000000, float 0.5)
  ret float %retval
; CHECK-NEXT: ret float 0x7FF0000000000000
}

define double @test_simplify10(double %x) {
; CHECK-LABEL: @test_simplify10(
  %retval = call double @pow(double 0xFFF0000000000000, double 0.5)
  ret double %retval
; CHECK-NEXT: ret double 0x7FF0000000000000
}

; Check pow(x, 1.0) -> x.

define float @test_simplify11(float %x) {
; CHECK-LABEL: @test_simplify11(
  %retval = call float @powf(float %x, float 1.0)
  ret float %retval
; CHECK-NEXT: ret float %x
}

define double @test_simplify12(double %x) {
; CHECK-LABEL: @test_simplify12(
  %retval = call double @pow(double %x, double 1.0)
  ret double %retval
; CHECK-NEXT: ret double %x
}

; Check pow(x, 2.0) -> x*x.

define float @test_simplify13(float %x) {
; CHECK-LABEL: @test_simplify13(
  %retval = call float @powf(float %x, float 2.0)
; CHECK-NEXT: [[SQUARE:%[a-z0-9]+]] = fmul float %x, %x
  ret float %retval
; CHECK-NEXT: ret float [[SQUARE]]
}

define double @test_simplify14(double %x) {
; CHECK-LABEL: @test_simplify14(
  %retval = call double @pow(double %x, double 2.0)
; CHECK-NEXT: [[SQUARE:%[a-z0-9]+]] = fmul double %x, %x
  ret double %retval
; CHECK-NEXT: ret double [[SQUARE]]
}

; Check pow(x, -1.0) -> 1.0/x.

define float @test_simplify15(float %x) {
; CHECK-LABEL: @test_simplify15(
  %retval = call float @powf(float %x, float -1.0)
; CHECK-NEXT: [[RECIPROCAL:%[a-z0-9]+]] = fdiv float 1.000000e+00, %x
  ret float %retval
; CHECK-NEXT: ret float [[RECIPROCAL]]
}

define double @test_simplify16(double %x) {
; CHECK-LABEL: @test_simplify16(
  %retval = call double @pow(double %x, double -1.0)
; CHECK-NEXT: [[RECIPROCAL:%[a-z0-9]+]] = fdiv double 1.000000e+00, %x
  ret double %retval
; CHECK-NEXT: ret double [[RECIPROCAL]]
}

declare double @llvm.pow.f64(double %Val, double %Power)
define double @test_simplify17(double %x) {
; CHECK-LABEL: @test_simplify17(
  %retval = call double @llvm.pow.f64(double %x, double 0.5)
; CHECK-NEXT: [[SQRT:%[a-z0-9]+]] = call double @sqrt(double %x)
; CHECK-NEXT: [[FABS:%[a-z0-9]+]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT: [[FCMP:%[a-z0-9]+]] = fcmp oeq double %x, 0xFFF0000000000000
; CHECK-NEXT: [[SELECT:%[a-z0-9]+]] = select i1 [[FCMP]], double 0x7FF0000000000000, double [[FABS]]
  ret double %retval
; CHECK-NEXT: ret double [[SELECT]]
}

; Check pow(10.0, x) -> __exp10(x) on OS X 10.9+ and iOS 7.0+.

define float @test_simplify18(float %x) {
; CHECK-LABEL: @test_simplify18(
  %retval = call float @powf(float 10.0, float %x)
; CHECK-EXP10: [[EXP10F:%[_a-z0-9]+]] = call float @__exp10f(float %x) [[NUW_RO:#[0-9]+]]
  ret float %retval
; CHECK-EXP10: ret float [[EXP10F]]
; CHECK-NO-EXP10: call float @powf
}

define double @test_simplify19(double %x) {
; CHECK-LABEL: @test_simplify19(
  %retval = call double @pow(double 10.0, double %x)
; CHECK-EXP10: [[EXP10:%[_a-z0-9]+]] = call double @__exp10(double %x) [[NUW_RO]]
  ret double %retval
; CHECK-EXP10: ret double [[EXP10]]
; CHECK-NO-EXP10: call double @pow
}

; CHECK: attributes [[NUW_RO]] = { nounwind readonly }

