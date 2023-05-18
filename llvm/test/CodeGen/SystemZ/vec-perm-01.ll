; Test vector splat.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 splat of the first element.
define <16 x i8> @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vrepb %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> zeroinitializer
  ret <16 x i8> %ret
}

; Test v16i8 splat of the last element.
define <16 x i8> @f2(<16 x i8> %val) {
; CHECK-LABEL: f2:
; CHECK: vrepb %v24, %v24, 15
; CHECK: br %r14
  %ret = shufflevector <16 x i8> %val, <16 x i8> undef,
                       <16 x i32> <i32 15, i32 15, i32 15, i32 15,
                                   i32 15, i32 15, i32 15, i32 15,
                                   i32 15, i32 15, i32 15, i32 15,
                                   i32 15, i32 15, i32 15, i32 15>
  ret <16 x i8> %ret
}

; Test v16i8 splat of an arbitrary element, using the second operand of
; the shufflevector.
define <16 x i8> @f3(<16 x i8> %val) {
; CHECK-LABEL: f3:
; CHECK: vrepb %v24, %v24, 4
; CHECK: br %r14
  %ret = shufflevector <16 x i8> undef, <16 x i8> %val,
                       <16 x i32> <i32 20, i32 20, i32 20, i32 20,
                                   i32 20, i32 20, i32 20, i32 20,
                                   i32 20, i32 20, i32 20, i32 20,
                                   i32 20, i32 20, i32 20, i32 20>
  ret <16 x i8> %ret
}

; Test v8i16 splat of the first element.
define <8 x i16> @f4(<8 x i16> %val) {
; CHECK-LABEL: f4:
; CHECK: vreph %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> zeroinitializer
  ret <8 x i16> %ret
}

; Test v8i16 splat of the last element.
define <8 x i16> @f5(<8 x i16> %val) {
; CHECK-LABEL: f5:
; CHECK: vreph %v24, %v24, 7
; CHECK: br %r14
  %ret = shufflevector <8 x i16> %val, <8 x i16> undef,
                       <8 x i32> <i32 7, i32 7, i32 7, i32 7,
                                  i32 7, i32 7, i32 7, i32 7>
  ret <8 x i16> %ret
}

; Test v8i16 splat of an arbitrary element, using the second operand of
; the shufflevector.
define <8 x i16> @f6(<8 x i16> %val) {
; CHECK-LABEL: f6:
; CHECK: vreph %v24, %v24, 2
; CHECK: br %r14
  %ret = shufflevector <8 x i16> undef, <8 x i16> %val,
                       <8 x i32> <i32 10, i32 10, i32 10, i32 10,
                                  i32 10, i32 10, i32 10, i32 10>
  ret <8 x i16> %ret
}

; Test v4i32 splat of the first element.
define <4 x i32> @f7(<4 x i32> %val) {
; CHECK-LABEL: f7:
; CHECK: vrepf %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> zeroinitializer
  ret <4 x i32> %ret
}

; Test v4i32 splat of the last element.
define <4 x i32> @f8(<4 x i32> %val) {
; CHECK-LABEL: f8:
; CHECK: vrepf %v24, %v24, 3
; CHECK: br %r14
  %ret = shufflevector <4 x i32> %val, <4 x i32> undef,
                       <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  ret <4 x i32> %ret
}

; Test v4i32 splat of an arbitrary element, using the second operand of
; the shufflevector.
define <4 x i32> @f9(<4 x i32> %val) {
; CHECK-LABEL: f9:
; CHECK: vrepf %v24, %v24, 1
; CHECK: br %r14
  %ret = shufflevector <4 x i32> undef, <4 x i32> %val,
                       <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  ret <4 x i32> %ret
}

; Test v2i64 splat of the first element.
define <2 x i64> @f10(<2 x i64> %val) {
; CHECK-LABEL: f10:
; CHECK: vrepg %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> zeroinitializer
  ret <2 x i64> %ret
}

; Test v2i64 splat of the last element.
define <2 x i64> @f11(<2 x i64> %val) {
; CHECK-LABEL: f11:
; CHECK: vrepg %v24, %v24, 1
; CHECK: br %r14
  %ret = shufflevector <2 x i64> %val, <2 x i64> undef,
                       <2 x i32> <i32 1, i32 1>
  ret <2 x i64> %ret
}

; Test v4f32 splat of the first element.
define <4 x float> @f12(<4 x float> %val) {
; CHECK-LABEL: f12:
; CHECK: vrepf %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val, <4 x float> undef,
                       <4 x i32> zeroinitializer
  ret <4 x float> %ret
}

; Test v4f32 splat of the last element.
define <4 x float> @f13(<4 x float> %val) {
; CHECK-LABEL: f13:
; CHECK: vrepf %v24, %v24, 3
; CHECK: br %r14
  %ret = shufflevector <4 x float> %val, <4 x float> undef,
                       <4 x i32> <i32 3, i32 3, i32 3, i32 3>
  ret <4 x float> %ret
}

; Test v4f32 splat of an arbitrary element, using the second operand of
; the shufflevector.
define <4 x float> @f14(<4 x float> %val) {
; CHECK-LABEL: f14:
; CHECK: vrepf %v24, %v24, 1
; CHECK: br %r14
  %ret = shufflevector <4 x float> undef, <4 x float> %val,
                       <4 x i32> <i32 5, i32 5, i32 5, i32 5>
  ret <4 x float> %ret
}

; Test v2f64 splat of the first element.
define <2 x double> @f15(<2 x double> %val) {
; CHECK-LABEL: f15:
; CHECK: vrepg %v24, %v24, 0
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val, <2 x double> undef,
                       <2 x i32> zeroinitializer
  ret <2 x double> %ret
}

; Test v2f64 splat of the last element.
define <2 x double> @f16(<2 x double> %val) {
; CHECK-LABEL: f16:
; CHECK: vrepg %v24, %v24, 1
; CHECK: br %r14
  %ret = shufflevector <2 x double> %val, <2 x double> undef,
                       <2 x i32> <i32 1, i32 1>
  ret <2 x double> %ret
}
