; Test vector insertion of constants.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 insertion into the first element.
define <16 x i8> @f1(<16 x i8> %val) {
; CHECK-LABEL: f1:
; CHECK: vleib %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 0, i32 0
  ret <16 x i8> %ret
}

; Test v16i8 insertion into the last element.
define <16 x i8> @f2(<16 x i8> %val) {
; CHECK-LABEL: f2:
; CHECK: vleib %v24, 100, 15
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 100, i32 15
  ret <16 x i8> %ret
}

; Test v16i8 insertion with the maximum signed value.
define <16 x i8> @f3(<16 x i8> %val) {
; CHECK-LABEL: f3:
; CHECK: vleib %v24, 127, 10
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 127, i32 10
  ret <16 x i8> %ret
}

; Test v16i8 insertion with the minimum signed value.
define <16 x i8> @f4(<16 x i8> %val) {
; CHECK-LABEL: f4:
; CHECK: vleib %v24, -128, 11
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 128, i32 11
  ret <16 x i8> %ret
}

; Test v16i8 insertion with the maximum unsigned value.
define <16 x i8> @f5(<16 x i8> %val) {
; CHECK-LABEL: f5:
; CHECK: vleib %v24, -1, 12
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 255, i32 12
  ret <16 x i8> %ret
}

; Test v16i8 insertion into a variable element.
define <16 x i8> @f6(<16 x i8> %val, i32 %index) {
; CHECK-LABEL: f6:
; CHECK-NOT: vleib
; CHECK: br %r14
  %ret = insertelement <16 x i8> %val, i8 0, i32 %index
  ret <16 x i8> %ret
}

; Test v8i16 insertion into the first element.
define <8 x i16> @f7(<8 x i16> %val) {
; CHECK-LABEL: f7:
; CHECK: vleih %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 0, i32 0
  ret <8 x i16> %ret
}

; Test v8i16 insertion into the last element.
define <8 x i16> @f8(<8 x i16> %val) {
; CHECK-LABEL: f8:
; CHECK: vleih %v24, 0, 7
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 0, i32 7
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the maximum signed value.
define <8 x i16> @f9(<8 x i16> %val) {
; CHECK-LABEL: f9:
; CHECK: vleih %v24, 32767, 4
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 32767, i32 4
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the minimum signed value.
define <8 x i16> @f10(<8 x i16> %val) {
; CHECK-LABEL: f10:
; CHECK: vleih %v24, -32768, 5
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 32768, i32 5
  ret <8 x i16> %ret
}

; Test v8i16 insertion with the maximum unsigned value.
define <8 x i16> @f11(<8 x i16> %val) {
; CHECK-LABEL: f11:
; CHECK: vleih %v24, -1, 6
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 65535, i32 6
  ret <8 x i16> %ret
}

; Test v8i16 insertion into a variable element.
define <8 x i16> @f12(<8 x i16> %val, i32 %index) {
; CHECK-LABEL: f12:
; CHECK-NOT: vleih
; CHECK: br %r14
  %ret = insertelement <8 x i16> %val, i16 0, i32 %index
  ret <8 x i16> %ret
}

; Test v4i32 insertion into the first element.
define <4 x i32> @f13(<4 x i32> %val) {
; CHECK-LABEL: f13:
; CHECK: vleif %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 0, i32 0
  ret <4 x i32> %ret
}

; Test v4i32 insertion into the last element.
define <4 x i32> @f14(<4 x i32> %val) {
; CHECK-LABEL: f14:
; CHECK: vleif %v24, 0, 3
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 0, i32 3
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the maximum value allowed by VLEIF.
define <4 x i32> @f15(<4 x i32> %val) {
; CHECK-LABEL: f15:
; CHECK: vleif %v24, 32767, 1
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 32767, i32 1
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the next value up.
define <4 x i32> @f16(<4 x i32> %val) {
; CHECK-LABEL: f16:
; CHECK-NOT: vleif
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 32768, i32 1
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the minimum value allowed by VLEIF.
define <4 x i32> @f17(<4 x i32> %val) {
; CHECK-LABEL: f17:
; CHECK: vleif %v24, -32768, 2
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 -32768, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion with the next value down.
define <4 x i32> @f18(<4 x i32> %val) {
; CHECK-LABEL: f18:
; CHECK-NOT: vleif
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 -32769, i32 2
  ret <4 x i32> %ret
}

; Test v4i32 insertion into a variable element.
define <4 x i32> @f19(<4 x i32> %val, i32 %index) {
; CHECK-LABEL: f19:
; CHECK-NOT: vleif
; CHECK: br %r14
  %ret = insertelement <4 x i32> %val, i32 0, i32 %index
  ret <4 x i32> %ret
}

; Test v2i64 insertion into the first element.
define <2 x i64> @f20(<2 x i64> %val) {
; CHECK-LABEL: f20:
; CHECK: vleig %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 0, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into the last element.
define <2 x i64> @f21(<2 x i64> %val) {
; CHECK-LABEL: f21:
; CHECK: vleig %v24, 0, 1
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 0, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the maximum value allowed by VLEIG.
define <2 x i64> @f22(<2 x i64> %val) {
; CHECK-LABEL: f22:
; CHECK: vleig %v24, 32767, 1
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 32767, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the next value up.
define <2 x i64> @f23(<2 x i64> %val) {
; CHECK-LABEL: f23:
; CHECK-NOT: vleig
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 32768, i32 1
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the minimum value allowed by VLEIG.
define <2 x i64> @f24(<2 x i64> %val) {
; CHECK-LABEL: f24:
; CHECK: vleig %v24, -32768, 0
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 -32768, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion with the next value down.
define <2 x i64> @f25(<2 x i64> %val) {
; CHECK-LABEL: f25:
; CHECK-NOT: vleig
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 -32769, i32 0
  ret <2 x i64> %ret
}

; Test v2i64 insertion into a variable element.
define <2 x i64> @f26(<2 x i64> %val, i32 %index) {
; CHECK-LABEL: f26:
; CHECK-NOT: vleig
; CHECK: br %r14
  %ret = insertelement <2 x i64> %val, i64 0, i32 %index
  ret <2 x i64> %ret
}

; Test v4f32 insertion of 0 into the first element.
define <4 x float> @f27(<4 x float> %val) {
; CHECK-LABEL: f27:
; CHECK: vleif %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float 0.0, i32 0
  ret <4 x float> %ret
}

; Test v4f32 insertion of 0 into the last element.
define <4 x float> @f28(<4 x float> %val) {
; CHECK-LABEL: f28:
; CHECK: vleif %v24, 0, 3
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float 0.0, i32 3
  ret <4 x float> %ret
}

; Test v4f32 insertion of a nonzero value.
define <4 x float> @f29(<4 x float> %val) {
; CHECK-LABEL: f29:
; CHECK-NOT: vleif
; CHECK: br %r14
  %ret = insertelement <4 x float> %val, float 1.0, i32 1
  ret <4 x float> %ret
}

; Test v2f64 insertion of 0 into the first element.
define <2 x double> @f30(<2 x double> %val) {
; CHECK-LABEL: f30:
; CHECK: vleig %v24, 0, 0
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double 0.0, i32 0
  ret <2 x double> %ret
}

; Test v2f64 insertion of 0 into the last element.
define <2 x double> @f31(<2 x double> %val) {
; CHECK-LABEL: f31:
; CHECK: vleig %v24, 0, 1
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double 0.0, i32 1
  ret <2 x double> %ret
}

; Test v2f64 insertion of a nonzero value.
define <2 x double> @f32(<2 x double> %val) {
; CHECK-LABEL: f32:
; CHECK-NOT: vleig
; CHECK: br %r14
  %ret = insertelement <2 x double> %val, double 1.0, i32 1
  ret <2 x double> %ret
}
