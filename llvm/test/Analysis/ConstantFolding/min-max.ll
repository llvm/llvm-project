; RUN: opt -instcombine -S -o - %s | FileCheck %s
; Tests that constant folding of min and max operations works as expected.

declare float @llvm.minnum.f32(float, float)
declare float @llvm.maxnum.f32(float, float)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)

declare float @llvm.minimum.f32(float, float)
declare float @llvm.maximum.f32(float, float)
declare <4 x float> @llvm.minimum.v4f32(<4 x float>, <4 x float>)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)

; CHECK: define float @minnum_float() {
define float @minnum_float() {
  ; CHECK-NEXT: ret float 5.000000e+00
  %1 = call float @llvm.minnum.f32(float 5.0, float 42.0)
  ret float %1
}

; Check that minnum constant folds to propagate non-NaN or smaller argument
; CHECK: define <4 x float> @minnum_float_vec() {
define <4 x float> @minnum_float_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0x7FF8000000000000, float 5.000000e+00,
  ; CHECK-SAME:                  float 4.200000e+01, float 5.000000e+00>
  %1 = call <4 x float> @llvm.minnum.v4f32(
    <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000, float 42., float 42.>,
    <4 x float> <float 0x7FF8000000000000, float 5., float 0x7FF8000000000000, float 5.>
  )
  ret <4 x float> %1
}

; Check that minnum constant folds to propagate one of its argument zeros
; CHECK: define <4 x float> @minnum_float_zeros_vec() {
define <4 x float> @minnum_float_zeros_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0.000000e+00, float {{-?}}0.000000e+00,
  ; CHECK-SAME:                  float {{-?}}0.000000e+00, float -0.000000e+00>
  %1 = call <4 x float> @llvm.minnum.v4f32(
    <4 x float> <float 0.0, float -0.0, float 0.0, float -0.0>,
    <4 x float> <float 0.0, float 0.0, float -0.0, float -0.0>
  )
  ret <4 x float> %1
}

; CHECK: define float @maxnum_float() {
define float @maxnum_float() {
  ; CHECK-NEXT: ret float 4.200000e+01
  %1 = call float @llvm.maxnum.f32(float 5.0, float 42.0)
  ret float %1
}

; Check that maxnum constant folds to propagate non-NaN or greater argument
; CHECK: define <4 x float> @maxnum_float_vec() {
define <4 x float> @maxnum_float_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0x7FF8000000000000, float 5.000000e+00,
  ; CHECK-SAME:                  float 4.200000e+01, float 4.200000e+01>
  %1 = call <4 x float> @llvm.maxnum.v4f32(
    <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000, float 42., float 42.>,
    <4 x float> <float 0x7FF8000000000000, float 5., float 0x7FF8000000000000, float 5.>
  )
  ret <4 x float> %1
}

; Check that maxnum constant folds to propagate one of its argument zeros
; CHECK: define <4 x float> @maxnum_float_zeros_vec() {
define <4 x float> @maxnum_float_zeros_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0.000000e+00, float {{-?}}0.000000e+00,
  ; CHECK-SAME:                  float {{-?}}0.000000e+00, float -0.000000e+00>
  %1 = call <4 x float> @llvm.maxnum.v4f32(
    <4 x float> <float 0.0, float -0.0, float 0.0, float -0.0>,
    <4 x float> <float 0.0, float 0.0, float -0.0, float -0.0>
  )
  ret <4 x float> %1
}

; CHECK: define float @minimum_float() {
define float @minimum_float() {
  ; CHECK-NEXT: ret float 5.000000e+00
  %1 = call float @llvm.minimum.f32(float 5.0, float 42.0)
  ret float %1
}

; Check that minimum propagates its NaN or smaller argument
; CHECK: define <4 x float> @minimum_float_vec() {
define <4 x float> @minimum_float_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000,
  ; CHECK-SAME:                  float 0x7FF8000000000000, float 5.000000e+00>
  %1 = call <4 x float> @llvm.minimum.v4f32(
    <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000, float 42., float 42.>,
    <4 x float> <float 0x7FF8000000000000, float 5., float 0x7FF8000000000000, float 5.>
  )
  ret <4 x float> %1
}

; Check that minimum treats -0.0 as smaller than 0.0 while constant folding
; CHECK: define <4 x float> @minimum_float_zeros_vec() {
define <4 x float> @minimum_float_zeros_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0.000000e+00, float -0.000000e+00,
  ; CHECK-SAME:                  float -0.000000e+00, float -0.000000e+00>
  %1 = call <4 x float> @llvm.minimum.v4f32(
    <4 x float> <float 0.0, float -0.0, float 0.0, float -0.0>,
    <4 x float> <float 0.0, float 0.0, float -0.0, float -0.0>
  )
  ret <4 x float> %1
}

; CHECK: define float @maximum_float() {
define float @maximum_float() {
  ; CHECK-NEXT: ret float 4.200000e+01
  %1 = call float @llvm.maximum.f32(float 5.0, float 42.0)
  ret float %1
}

; Check that maximum propagates its NaN or greater argument
; CHECK: define <4 x float> @maximum_float_vec() {
define <4 x float> @maximum_float_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000,
  ; CHECK-SAME:                  float 0x7FF8000000000000, float 4.200000e+01>
  %1 = call <4 x float> @llvm.maximum.v4f32(
    <4 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000, float 42., float 42.>,
    <4 x float> <float 0x7FF8000000000000, float 5., float 0x7FF8000000000000, float 5.>
  )
  ret <4 x float> %1
}

; Check that maximum treats -0.0 as smaller than 0.0 while constant folding
; CHECK: define <4 x float> @maximum_float_zeros_vec() {
define <4 x float> @maximum_float_zeros_vec() {
  ; CHECK-NEXT: ret <4 x float> <float 0.000000e+00, float 0.000000e+00,
  ; CHECK-SAME:                  float 0.000000e+00, float -0.000000e+00>
  %1 = call <4 x float> @llvm.maximum.v4f32(
    <4 x float> <float 0.0, float -0.0, float 0.0, float -0.0>,
    <4 x float> <float 0.0, float 0.0, float -0.0, float -0.0>
  )
  ret <4 x float> %1
}
