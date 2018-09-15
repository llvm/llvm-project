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
declare <2 x float> @llvm.pow.v2f32(<2 x float>, <2 x float>) nounwind readonly
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>) nounwind readonly

; Check pow(1.0, x) -> 1.0.

define float @test_simplify1(float %x) {
; CHECK-LABEL: @test_simplify1(
  %retval = call float @powf(float 1.0, float %x)
  ret float %retval
; CHECK-NEXT: ret float 1.000000e+00
}

define <2 x float> @test_simplify1v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify1v(
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 1.0, float 1.0>, <2 x float> %x)
  ret <2 x float> %retval
; CHECK-NEXT: ret <2 x float> <float 1.000000e+00, float 1.000000e+00>
}

define double @test_simplify2(double %x) {
; CHECK-LABEL: @test_simplify2(
  %retval = call double @pow(double 1.0, double %x)
  ret double %retval
; CHECK-NEXT: ret double 1.000000e+00
}

define <2 x double> @test_simplify2v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify2v(
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 1.0, double 1.0>, <2 x double> %x)
  ret <2 x double> %retval
; CHECK-NEXT: ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
}

; Check pow(2.0, x) -> exp2(x).

define float @test_simplify3(float %x) {
; CHECK-LABEL: @test_simplify3(
  %retval = call float @powf(float 2.0, float %x)
; CHECK-NEXT: [[EXP2F:%[a-z0-9]+]] = call float @llvm.exp2.f32(float %x)
  ret float %retval
; CHECK-NEXT: ret float [[EXP2F]]
}

define <2 x float> @test_simplify3v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify3v(
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 2.0, float 2.0>, <2 x float> %x)
; CHECK-NEXT: [[EXP2F:%[a-z0-9]+]] = call <2 x float> @llvm.exp2.v2f32(<2 x float> %x)
  ret <2 x float> %retval
; CHECK-NEXT: ret <2 x float> [[EXP2F]]
}

define double @test_simplify4(double %x) {
; CHECK-LABEL: @test_simplify4(
  %retval = call double @pow(double 2.0, double %x)
; CHECK-NEXT: [[EXP2:%[a-z0-9]+]] = call double @llvm.exp2.f64(double %x)
  ret double %retval
; CHECK-NEXT: ret double [[EXP2]]
}

define <2 x double> @test_simplify4v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify4v(
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 2.0, double 2.0>, <2 x double> %x)
; CHECK-NEXT: [[EXP2:%[a-z0-9]+]] = call <2 x double> @llvm.exp2.v2f64(<2 x double> %x)
  ret <2 x double> %retval
; CHECK-NEXT: ret <2 x double> [[EXP2]]
}

; Check pow(x, 0.0) -> 1.0.

define float @test_simplify5(float %x) {
; CHECK-LABEL: @test_simplify5(
  %retval = call float @powf(float %x, float 0.0)
  ret float %retval
; CHECK-NEXT: ret float 1.000000e+00
}

define <2 x float> @test_simplify5v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify5v(
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 0.0, float 0.0>)
  ret <2 x float> %retval
; CHECK-NEXT: %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> zeroinitializer)
}

define double @test_simplify6(double %x) {
; CHECK-LABEL: @test_simplify6(
  %retval = call double @pow(double %x, double 0.0)
  ret double %retval
; CHECK-NEXT: ret double 1.000000e+00
}

define <2 x double> @test_simplify6v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify6v(
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 0.0, double 0.0>)
  ret <2 x double> %retval
; CHECK-NEXT: %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> zeroinitializer)
}

; Check pow(x, 0.5) -> fabs(sqrt(x)), where x != -infinity.

define float @test_simplify7(float %x) {
; CHECK-LABEL: @test_simplify7(
  %retval = call float @powf(float %x, float 0.5)
; CHECK-NEXT: [[SQRTF:%[a-z0-9]+]] = call float @sqrtf(float %x) [[$NUW_RO:#[0-9]+]]
; CHECK-NEXT: [[FABSF:%[a-z0-9]+]] = call float @llvm.fabs.f32(float [[SQRTF]])
; CHECK-NEXT: [[FCMP:%[a-z0-9]+]] = fcmp oeq float %x, 0xFFF0000000000000
; CHECK-NEXT: [[SELECT:%[a-z0-9]+]] = select i1 [[FCMP]], float 0x7FF0000000000000, float [[FABSF]]
  ret float %retval
; CHECK-NEXT: ret float [[SELECT]]
}

define double @test_simplify8(double %x) {
; CHECK-LABEL: @test_simplify8(
  %retval = call double @pow(double %x, double 0.5)
; CHECK-NEXT: [[SQRT:%[a-z0-9]+]] = call double @sqrt(double %x) [[$NUW_RO]]
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

define <2 x float> @test_simplify11v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify11v(
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 1.0, float 1.0>)
  ret <2 x float> %retval
; CHECK-NEXT: %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 1.000000e+00, float 1.000000e+00>)
}

define double @test_simplify12(double %x) {
; CHECK-LABEL: @test_simplify12(
  %retval = call double @pow(double %x, double 1.0)
  ret double %retval
; CHECK-NEXT: ret double %x
}

define <2 x double> @test_simplify12v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify12v(
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 1.0, double 1.0>)
  ret <2 x double> %retval
; CHECK-NEXT: %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 1.000000e+00, double 1.000000e+00>)
}

; Check pow(x, 2.0) -> x*x.

define float @pow2_strict(float %x) {
; CHECK-LABEL: @pow2_strict(
; CHECK-NEXT:    [[POW2:%.*]] = fmul float %x, %x
; CHECK-NEXT:    ret float [[POW2]]
;
  %r = call float @powf(float %x, float 2.0)
  ret float %r
}

define <2 x float> @pow2_strictv(<2 x float> %x) {
; CHECK-LABEL: @pow2_strictv(
; CHECK-NEXT:    [[POW2:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 2.000000e+00, float 2.000000e+00>)
; CHECK-NEXT:    ret <2 x float> [[POW2]]
;
  %r = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 2.0, float 2.0>)
  ret <2 x float> %r
}

define double @pow2_double_strict(double %x) {
; CHECK-LABEL: @pow2_double_strict(
; CHECK-NEXT:    [[POW2:%.*]] = fmul double %x, %x
; CHECK-NEXT:    ret double [[POW2]]
;
  %r = call double @pow(double %x, double 2.0)
  ret double %r
}
define <2 x double> @pow2_double_strictv(<2 x double> %x) {
; CHECK-LABEL: @pow2_double_strictv(
; CHECK-NEXT:    [[POW2:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 2.000000e+00, double 2.000000e+00>)
; CHECK-NEXT:    ret <2 x double> [[POW2]]
;
  %r = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 2.0, double 2.0>)
  ret <2 x double> %r
}

; Don't drop the FMF - PR35601 ( https://bugs.llvm.org/show_bug.cgi?id=35601 )

define float @pow2_fast(float %x) {
; CHECK-LABEL: @pow2_fast(
; CHECK-NEXT:    [[POW2:%.*]] = fmul fast float %x, %x
; CHECK-NEXT:    ret float [[POW2]]
;
  %r = call fast float @powf(float %x, float 2.0)
  ret float %r
}

; Check pow(x, -1.0) -> 1.0/x.

define float @pow_neg1_strict(float %x) {
; CHECK-LABEL: @pow_neg1_strict(
; CHECK-NEXT:    [[POWRECIP:%.*]] = fdiv float 1.000000e+00, %x
; CHECK-NEXT:    ret float [[POWRECIP]]
;
  %r = call float @powf(float %x, float -1.0)
  ret float %r
}

define <2 x float> @pow_neg1_strictv(<2 x float> %x) {
; CHECK-LABEL: @pow_neg1_strictv(
; CHECK-NEXT:    [[POWRECIP:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float -1.000000e+00, float -1.000000e+00>)
; CHECK-NEXT:    ret <2 x float> [[POWRECIP]]
;
  %r = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float -1.0, float -1.0>)
  ret <2 x float> %r
}

define double @pow_neg1_double_fast(double %x) {
; CHECK-LABEL: @pow_neg1_double_fast(
; CHECK-NEXT:    [[POWRECIP:%.*]] = fdiv fast double 1.000000e+00, %x
; CHECK-NEXT:    ret double [[POWRECIP]]
;
  %r = call fast double @pow(double %x, double -1.0)
  ret double %r
}

define <2 x double> @pow_neg1_double_fastv(<2 x double> %x) {
; CHECK-LABEL: @pow_neg1_double_fastv(
; CHECK-NEXT:    [[POWRECIP:%.*]] = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double -1.000000e+00, double -1.000000e+00>)
; CHECK-NEXT:    ret <2 x double> [[POWRECIP]]
;
  %r = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double -1.0, double -1.0>)
  ret <2 x double> %r
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
; CHECK-EXP10: [[EXP10F:%[_a-z0-9]+]] = call float @__exp10f(float %x) [[$NUW_RO:#[0-9]+]]
  ret float %retval
; CHECK-EXP10: ret float [[EXP10F]]
; CHECK-NO-EXP10: call float @powf
}

define double @test_simplify19(double %x) {
; CHECK-LABEL: @test_simplify19(
  %retval = call double @pow(double 10.0, double %x)
; CHECK-EXP10: [[EXP10:%[_a-z0-9]+]] = call double @__exp10(double %x) [[$NUW_RO]]
  ret double %retval
; CHECK-EXP10: ret double [[EXP10]]
; CHECK-NO-EXP10: call double @pow
}

; CHECK: attributes [[$NUW_RO]] = { nounwind readonly }

