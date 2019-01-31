; Test that the pow library call simplifier works correctly.
;
; RUN: opt -instcombine -S < %s                                  | FileCheck %s --check-prefixes=CHECK,ANY
; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.9 | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios7.0        | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=x86_64-apple-macosx10.8 | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-ios6.0        | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=x86_64-netbsd           | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-NO-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-tvos9.0       | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-EXP10
; RUN: opt -instcombine -S < %s -mtriple=arm-apple-watchos2.0    | FileCheck %s --check-prefixes=CHECK,ANY,CHECK-EXP10
; rdar://7251832
; RUN: opt -instcombine -S < %s -mtriple=x86_64-pc-windows-msvc  | FileCheck %s --check-prefixes=CHECK,WIN,CHECK-NO-EXP10

; NOTE: The readonly attribute on the pow call should be preserved
; in the cases below where pow is transformed into another function call.

declare float @powf(float, float) nounwind readonly
declare double @pow(double, double) nounwind readonly
declare double @llvm.pow.f64(double, double)
declare <2 x float> @llvm.pow.v2f32(<2 x float>, <2 x float>) nounwind readonly
declare <2 x double> @llvm.pow.v2f64(<2 x double>, <2 x double>) nounwind readonly

; Check pow(1.0, x) -> 1.0.

define float @test_simplify1(float %x) {
; CHECK-LABEL: @test_simplify1(
; CHECK-NEXT:  ret float 1.000000e+00
;
  %retval = call float @powf(float 1.0, float %x)
  ret float %retval
}

define <2 x float> @test_simplify1v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify1v(
; ANY-NEXT:    ret <2 x float> <float 1.000000e+00, float 1.000000e+00>
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 1.000000e+00, float 1.000000e+00>, <2 x float> %x)
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 1.0, float 1.0>, <2 x float> %x)
  ret <2 x float> %retval
}

define double @test_simplify2(double %x) {
; CHECK-LABEL: @test_simplify2(
; CHECK-NEXT:  ret double 1.000000e+00
;
  %retval = call double @pow(double 1.0, double %x)
  ret double %retval
}

define <2 x double> @test_simplify2v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify2v(
; ANY-NEXT:    ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 1.000000e+00, double 1.000000e+00>, <2 x double> %x)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 1.0, double 1.0>, <2 x double> %x)
  ret <2 x double> %retval
}

; Check pow(2.0 ** n, x) -> exp2(n * x).

define float @test_simplify3(float %x) {
; CHECK-LABEL: @test_simplify3(
; ANY-NEXT:    [[EXP2F:%.*]] = call float @exp2f(float [[X:%.*]]) [[NUW_RO:#[0-9]+]]
; ANY-NEXT:    ret float [[EXP2F]]
; WIN-NEXT:    [[POW:%.*]] = call float @powf(float 2.000000e+00, float [[X:%.*]])
; WIN-NEXT:    ret float [[POW]]
;
  %retval = call float @powf(float 2.0, float %x)
  ret float %retval
}

define double @test_simplify3n(double %x) {
; CHECK-LABEL: @test_simplify3n(
; ANY-NEXT:    [[MUL:%.*]] = fmul double [[X:%.*]], -2.000000e+00
; ANY-NEXT:    [[EXP2:%.*]] = call double @exp2(double [[MUL]]) [[NUW_RO]]
; ANY-NEXT:    ret double [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call double @pow(double 2.500000e-01, double [[X:%.*]])
; WIN-NEXT:    ret double [[POW]]
;
  %retval = call double @pow(double 0.25, double %x)
  ret double %retval
}

define <2 x float> @test_simplify3v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify3v(
; ANY-NEXT:    [[EXP2:%.*]] = call <2 x float> @llvm.exp2.v2f32(<2 x float> [[X:%.*]])
; ANY-NEXT:    ret <2 x float> [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 2.000000e+00, float 2.000000e+00>, <2 x float> [[X:%.*]])
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 2.0, float 2.0>, <2 x float> %x)
  ret <2 x float> %retval
}

define <2 x double> @test_simplify3vn(<2 x double> %x) {
; CHECK-LABEL: @test_simplify3vn(
; ANY-NEXT:    [[MUL:%.*]] = fmul <2 x double> [[X:%.*]], <double 2.000000e+00, double 2.000000e+00>
; ANY-NEXT:    [[EXP2:%.*]] = call <2 x double> @llvm.exp2.v2f64(<2 x double> [[MUL]])
; ANY-NEXT:    ret <2 x double> [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 4.000000e+00, double 4.000000e+00>, <2 x double> %x)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 4.0, double 4.0>, <2 x double> %x)
  ret <2 x double> %retval
}

define double @test_simplify4(double %x) {
; CHECK-LABEL: @test_simplify4(
; ANY-NEXT:    [[EXP2:%.*]] = call double @exp2(double [[X:%.*]]) [[NUW_RO]]
; ANY-NEXT:    ret double [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call double @pow(double 2.000000e+00, double [[X:%.*]])
; WIN-NEXT:    ret double [[POW]]
;
  %retval = call double @pow(double 2.0, double %x)
  ret double %retval
}

define float @test_simplify4n(float %x) {
; CHECK-LABEL: @test_simplify4n(
; ANY-NEXT:    [[MUL:%.*]] = fmul float [[X:%.*]], 3.000000e+00
; ANY-NEXT:    [[EXP2F:%.*]] = call float @exp2f(float [[MUL]]) [[NUW_RO]]
; ANY-NEXT:    ret float [[EXP2F]]
; WIN-NEXT:    [[POW:%.*]] = call float @powf(float 8.000000e+00, float [[X:%.*]])
; WIN-NEXT:    ret float [[POW]]
;
  %retval = call float @powf(float 8.0, float %x)
  ret float %retval
}

define <2 x double> @test_simplify4v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify4v(
; ANY-NEXT:    [[EXP2:%.*]] = call <2 x double> @llvm.exp2.v2f64(<2 x double> [[X:%.*]])
; ANY-NEXT:    ret <2 x double> [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 2.000000e+00, double 2.000000e+00>, <2 x double> [[X:%.*]])
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> <double 2.0, double 2.0>, <2 x double> %x)
  ret <2 x double> %retval
}

define <2 x float> @test_simplify4vn(<2 x float> %x) {
; CHECK-LABEL: @test_simplify4vn(
; ANY-NEXT:    [[MUL:%.*]] = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, [[X:%.*]]
; ANY-NEXT:    [[EXP2:%.*]] = call <2 x float> @llvm.exp2.v2f32(<2 x float> [[MUL]])
; ANY-NEXT:    ret <2 x float> [[EXP2]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 5.000000e-01, float 5.000000e-01>, <2 x float> [[X:%.*]])
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> <float 0.5, float 0.5>, <2 x float> %x)
  ret <2 x float> %retval
}

; Check pow(x, 0.0) -> 1.0.

define float @test_simplify5(float %x) {
; CHECK-LABEL: @test_simplify5(
; CHECK-NEXT:  ret float 1.000000e+00
;
  %retval = call float @powf(float %x, float 0.0)
  ret float %retval
}

define <2 x float> @test_simplify5v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify5v(
; ANY-NEXT:    ret <2 x float> <float 1.000000e+00, float 1.000000e+00>
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[X:%.*]], <2 x float> zeroinitializer)
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 0.0, float 0.0>)
  ret <2 x float> %retval
}

define double @test_simplify6(double %x) {
; CHECK-LABEL: @test_simplify6(
; CHECK-NEXT:  ret double 1.000000e+00
;
  %retval = call double @pow(double %x, double 0.0)
  ret double %retval
}

define <2 x double> @test_simplify6v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify6v(
; ANY-NEXT:    ret <2 x double> <double 1.000000e+00, double 1.000000e+00>
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> [[X:%.*]], <2 x double> zeroinitializer)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 0.0, double 0.0>)
  ret <2 x double> %retval
}

; Check pow(x, 0.5) -> fabs(sqrt(x)), where x != -infinity.

define float @test_simplify7(float %x) {
; CHECK-LABEL: @test_simplify7(
; ANY-NEXT:    [[SQRTF:%.*]] = call float @sqrtf(float [[X:%.*]]) [[NUW_RO]]
; WIN-NEXT:    [[SQRTF:%.*]] = call float @sqrtf(float [[X:%.*]]) [[NUW_RO:#[0-9]+]]
; CHECK-NEXT:  [[ABS:%.*]] = call float @llvm.fabs.f32(float [[SQRTF]])
; CHECK-NEXT:  [[ISINF:%.*]] = fcmp oeq float [[X]], 0xFFF0000000000000
; CHECK-NEXT:  [[TMP1:%.*]] = select i1 [[ISINF]], float 0x7FF0000000000000, float [[ABS]]
; CHECK-NEXT:  ret float [[TMP1]]
;
  %retval = call float @powf(float %x, float 0.5)
  ret float %retval
}

define double @test_simplify8(double %x) {
; CHECK-LABEL: @test_simplify8(
; CHECK-NEXT:  [[SQRT:%.*]] = call double @sqrt(double [[X:%.*]]) [[NUW_RO]]
; CHECK-NEXT:  [[ABS:%.*]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT:  [[ISINF:%.*]] = fcmp oeq double [[X]], 0xFFF0000000000000
; CHECK-NEXT:  [[TMP1:%.*]] = select i1 [[ISINF]], double 0x7FF0000000000000, double [[ABS]]
; CHECK-NEXT:  ret double [[TMP1]]
;
  %retval = call double @pow(double %x, double 0.5)
  ret double %retval
}

; Check pow(-infinity, 0.5) -> +infinity.

define float @test_simplify9(float %x) {
; CHECK-LABEL: @test_simplify9(
; CHECK-NEXT:  ret float 0x7FF0000000000000
;
  %retval = call float @powf(float 0xFFF0000000000000, float 0.5)
  ret float %retval
}

define double @test_simplify10(double %x) {
; CHECK-LABEL: @test_simplify10(
; CHECK-NEXT:  ret double 0x7FF0000000000000
;
  %retval = call double @pow(double 0xFFF0000000000000, double 0.5)
  ret double %retval
}

; Check pow(x, 1.0) -> x.

define float @test_simplify11(float %x) {
; CHECK-LABEL: @test_simplify11(
; CHECK-NEXT:  ret float [[X:%.*]]
;
  %retval = call float @powf(float %x, float 1.0)
  ret float %retval
}

define <2 x float> @test_simplify11v(<2 x float> %x) {
; CHECK-LABEL: @test_simplify11v(
; ANY-NEXT:    ret <2 x float> [[X:%.*]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[X:%.*]], <2 x float> <float 1.000000e+00, float 1.000000e+00>)
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %retval = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 1.0, float 1.0>)
  ret <2 x float> %retval
}

define double @test_simplify12(double %x) {
; CHECK-LABEL: @test_simplify12(
; CHECK-NEXT:  ret double [[X:%.*]]
;
  %retval = call double @pow(double %x, double 1.0)
  ret double %retval
}

define <2 x double> @test_simplify12v(<2 x double> %x) {
; CHECK-LABEL: @test_simplify12v(
; ANY-NEXT:    ret <2 x double> [[X:%.*]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> [[X:%.*]], <2 x double> <double 1.000000e+00, double 1.000000e+00>)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %retval = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 1.0, double 1.0>)
  ret <2 x double> %retval
}

; Check pow(x, 2.0) -> x*x.

define float @pow2_strict(float %x) {
; CHECK-LABEL: @pow2_strict(
; CHECK-NEXT:  [[SQUARE:%.*]] = fmul float [[X:%.*]], [[X]]
; CHECK-NEXT:  ret float [[SQUARE]]
;
  %r = call float @powf(float %x, float 2.0)
  ret float %r
}

define <2 x float> @pow2_strictv(<2 x float> %x) {
; CHECK-LABEL: @pow2_strictv(
; ANY-NEXT:    [[SQUARE:%.*]] = fmul <2 x float> [[X:%.*]], [[X]]
; ANY-NEXT:    ret <2 x float> [[SQUARE]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[X:%.*]], <2 x float> <float 2.000000e+00, float 2.000000e+00>)
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %r = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float 2.0, float 2.0>)
  ret <2 x float> %r
}

define double @pow2_double_strict(double %x) {
; CHECK-LABEL: @pow2_double_strict(
; CHECK-NEXT:  [[SQUARE:%.*]] = fmul double [[X:%.*]], [[X]]
; CHECK-NEXT:  ret double [[SQUARE]]
;
  %r = call double @pow(double %x, double 2.0)
  ret double %r
}

define <2 x double> @pow2_double_strictv(<2 x double> %x) {
; CHECK-LABEL: @pow2_double_strictv(
; ANY-NEXT:    [[SQUARE:%.*]] = fmul <2 x double> [[X:%.*]], [[X]]
; ANY-NEXT:    ret <2 x double> [[SQUARE]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x double> @llvm.pow.v2f64(<2 x double> [[X:%.*]], <2 x double> <double 2.000000e+00, double 2.000000e+00>)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %r = call <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double 2.0, double 2.0>)
  ret <2 x double> %r
}

; Don't drop the FMF - PR35601 ( https://bugs.llvm.org/show_bug.cgi?id=35601 )

define float @pow2_fast(float %x) {
; CHECK-LABEL: @pow2_fast(
; CHECK-NEXT:  [[SQUARE:%.*]] = fmul fast float [[X:%.*]], [[X]]
; CHECK-NEXT:  ret float [[SQUARE]]
;
  %r = call fast float @powf(float %x, float 2.0)
  ret float %r
}

; Check pow(x, -1.0) -> 1.0/x.

define float @pow_neg1_strict(float %x) {
; CHECK-LABEL: @pow_neg1_strict(
; CHECK-NEXT:  [[RECIPROCAL:%.*]] = fdiv float 1.000000e+00, [[X:%.*]]
; CHECK-NEXT:  ret float [[RECIPROCAL]]
;
  %r = call float @powf(float %x, float -1.0)
  ret float %r
}

define <2 x float> @pow_neg1_strictv(<2 x float> %x) {
; CHECK-LABEL: @pow_neg1_strictv(
; ANY-NEXT:    [[RECIPROCAL:%.*]] = fdiv <2 x float> <float 1.000000e+00, float 1.000000e+00>, [[X:%.*]]
; ANY-NEXT:    ret <2 x float> [[RECIPROCAL]]
; WIN-NEXT:    [[POW:%.*]] = call <2 x float> @llvm.pow.v2f32(<2 x float> [[X:%.*]], <2 x float> <float -1.000000e+00, float -1.000000e+00>)
; WIN-NEXT:    ret <2 x float> [[POW]]
;
  %r = call <2 x float> @llvm.pow.v2f32(<2 x float> %x, <2 x float> <float -1.0, float -1.0>)
  ret <2 x float> %r
}

define double @pow_neg1_double_fast(double %x) {
; CHECK-LABEL: @pow_neg1_double_fast(
; CHECK-NEXT:  [[RECIPROCAL:%.*]] = fdiv fast double 1.000000e+00, [[X:%.*]]
; CHECK-NEXT:  ret double [[RECIPROCAL]]
;
  %r = call fast double @pow(double %x, double -1.0)
  ret double %r
}

define <2 x double> @pow_neg1_double_fastv(<2 x double> %x) {
; CHECK-LABEL: @pow_neg1_double_fastv(
; ANY-NEXT:    [[RECIPROCAL:%.*]] = fdiv fast <2 x double> <double 1.000000e+00, double 1.000000e+00>, [[X:%.*]]
; ANY-NEXT:    ret <2 x double> [[RECIPROCAL]]
; WIN-NEXT:    [[POW:%.*]] = call fast <2 x double> @llvm.pow.v2f64(<2 x double> [[X:%.*]], <2 x double> <double -1.000000e+00, double -1.000000e+00>)
; WIN-NEXT:    ret <2 x double> [[POW]]
;
  %r = call fast <2 x double> @llvm.pow.v2f64(<2 x double> %x, <2 x double> <double -1.0, double -1.0>)
  ret <2 x double> %r
}

define double @test_simplify17(double %x) {
; CHECK-LABEL: @test_simplify17(
; CHECK-NEXT:  [[SQRT:%.*]] = call double @llvm.sqrt.f64(double [[X:%.*]])
; CHECK-NEXT:  [[ABS:%.*]] = call double @llvm.fabs.f64(double [[SQRT]])
; CHECK-NEXT:  [[ISINF:%.*]] = fcmp oeq double [[X]], 0xFFF0000000000000
; CHECK-NEXT:  [[TMP1:%.*]] = select i1 [[ISINF]], double 0x7FF0000000000000, double [[ABS]]
; CHECK-NEXT:  ret double [[TMP1]]
;
  %retval = call double @llvm.pow.f64(double %x, double 0.5)
  ret double %retval
}

; Check pow(10.0, x) -> __exp10(x) on OS X 10.9+ and iOS 7.0+.

define float @test_simplify18(float %x) {
; CHECK-LABEL:          @test_simplify18(
; CHECK-EXP10-NEXT:     [[__EXP10F:%.*]] = call float @__exp10f(float [[X:%.*]]) [[NUW_RO]]
; CHECK-EXP10-NEXT:     ret float [[__EXP10F]]
; CHECK-NO-EXP10-NEXT:  [[RETVAL:%.*]] = call float @powf(float 1.000000e+01, float [[X:%.*]])
; CHECK-NO-EXP10-NEXT:  ret float [[RETVAL]]
;
  %retval = call float @powf(float 10.0, float %x)
  ret float %retval
}

define double @test_simplify19(double %x) {
; CHECK-LABEL:          @test_simplify19(
; CHECK-EXP10-NEXT:     [[__EXP10:%.*]] = call double @__exp10(double [[X:%.*]]) [[NUW_RO]]
; CHECK-EXP10-NEXT:     ret double [[__EXP10]]
; CHECK-NO-EXP10-NEXT:  [[RETVAL:%.*]] = call double @pow(double 1.000000e+01, double [[X:%.*]])
; CHECK-NO-EXP10-NEXT:  ret double [[RETVAL]]
;
  %retval = call double @pow(double 10.0, double %x)
  ret double %retval
}

; CHECK: attributes [[NUW_RO]] = { nounwind readonly }
