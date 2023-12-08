; Test vector insertion of register variables.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 insertion into the first element.
define <16 x i8> @f1(<16 x i8> %val, i8 %element) {
; CHECK-LABEL: f1:
; CHECK: vlvgb %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 %element, i32 0
  ret <16 x i8> %ret
}

; Test v16i8 insertion into the last element.
define <16 x i8> @f2(<16 x i8> %val, i8 %element) {
; CHECK-LABEL: f2:
; CHECK: vlvgb %v24, %r2, 15
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 %element, i32 15
  ret <16 x i8> %ret
}

; Test v16i8 insertion into a variable element.
define <16 x i8> @f3(<16 x i8> %val, i8 %element, i32 %index) {
; CHECK-LABEL: f3:
; CHECK: vlvgb %v24, %r2, 0(%r3)
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 %element, i32 %index
  ret <16 x i8> %ret
}

; Test v8i16 insertion into the first element.
define <8 x i16> @f4(<8 x i16> %val, i16 %element) {
; CHECK-LABEL: f4:
; CHECK: vlvgh %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 %element, i32 0
  ret <8 x i16> %ret
}

; Test v8i16 insertion into the last element.
define <8 x i16> @f5(<8 x i16> %val, i16 %element) {
; CHECK-LABEL: f5:
; CHECK: vlvgh %v24, %r2, 7
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 %element, i32 7
  ret <8 x i16> %ret
}

; Test v8i16 insertion into a variable element.
define <8 x i16> @f6(<8 x i16> %val, i16 %element, i32 %index) {
; CHECK-LABEL: f6:
; CHECK: vlvgh %v24, %r2, 0(%r3)
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 %element, i32 %index
  ret <8 x i16> %ret
}

; Test v4i32 insertion into the first element.
define <4 x i32> @f7(<4 x i32> %val, i32 %element) {
; CHECK-LABEL: f7:
; CHECK: vlvgf %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 %element, i32 0
  ret <4 x i32> %ret
}

; Test v4i32 insertion into the last element.
define <4 x i32> @f8(<4 x i32> %val, i32 %element) {
; CHECK-LABEL: f8:
; CHECK: vlvgf %v24, %r2, 3
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 %element, i32 3
  ret <4 x i32> %ret
}

; Test v4i32 insertion into a variable element.
define <4 x i32> @f9(<4 x i32> %val, i32 %element, i32 %index) {
; CHECK-LABEL: f9:
; CHECK: vlvgf %v24, %r2, 0(%r3)
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 %element, i32 %index
  ret <4 x i32> %ret
}

; Test v2i64 insertion into the first element.
define <2 x i64> @f10(<2 x i64> %val, i64 %element) {
; CHECK-LABEL: f10:
; CHECK: vlvgg %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 %element, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into the last element.
define <2 x i64> @f11(<2 x i64> %val, i64 %element) {
; CHECK-LABEL: f11:
; CHECK: vlvgg %v24, %r2, 1
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 %element, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion into a variable element.
define <2 x i64> @f12(<2 x i64> %val, i64 %element, i32 %index) {
; CHECK-LABEL: f12:
; CHECK: vlvgg %v24, %r2, 0(%r3)
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 %element, i32 %index
  ret <2 x i64> %ret
}

; Test v4f32 insertion into the first element.
define <4 x float> @f13(<4 x float> %val, float %element) {
; CHECK-LABEL: f13:
; CHECK: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: vlvgf %v24, [[REG]], 0
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float %element, i32 0
  ret <4 x float> %ret
}

; Test v4f32 insertion into the last element.
define <4 x float> @f14(<4 x float> %val, float %element) {
; CHECK-LABEL: f14:
; CHECK: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: vlvgf %v24, [[REG]], 3
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float %element, i32 3
  ret <4 x float> %ret
}

; Test v4f32 insertion into a variable element.
define <4 x float> @f15(<4 x float> %val, float %element, i32 %index) {
; CHECK-LABEL: f15:
; CHECK: vlgvf [[REG:%r[0-5]]], %v0, 0
; CHECK: vlvgf %v24, [[REG]], 0(%r2)
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float %element, i32 %index
  ret <4 x float> %ret
}

; Test v2f64 insertion into the first element.
define <2 x double> @f16(<2 x double> %val, double %element) {
; CHECK-LABEL: f16:
; CHECK: vpdi %v24, %v0, %v24, 1
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double %element, i32 0
  ret <2 x double> %ret
}

; Test v2f64 insertion into the last element.
define <2 x double> @f17(<2 x double> %val, double %element) {
; CHECK-LABEL: f17:
; CHECK: vpdi %v24, %v24, %v0, 0
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double %element, i32 1
  ret <2 x double> %ret
}

; Test v2f64 insertion into a variable element.
define <2 x double> @f18(<2 x double> %val, double %element, i32 %index) {
; CHECK-LABEL: f18:
; CHECK: lgdr [[REG:%r[0-5]]], %f0
; CHECK: vlvgg %v24, [[REG]], 0(%r2)
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double %element, i32 %index
  ret <2 x double> %ret
}

; Test v16i8 insertion into a variable element plus one.
define <16 x i8> @f19(<16 x i8> %val, i8 %element, i32 %index) {
; CHECK-LABEL: f19:
; CHECK: vlvgb %v24, %r2, 1(%r3)
; CHECK: br %r14
  %add = add i32 %index, 1
  %ret = insertelement <16 x i8> %val, i8 %element, i32 %add
  ret <16 x i8> %ret
}
