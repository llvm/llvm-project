; Test scalar_to_vector expansion.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8.
define <16 x i8> @f1(i8 %val) {
; CHECK-LABEL: f1:
; CHECK: vlvgb %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <16 x i8> undef, i8 %val, i32 0
  ret <16 x i8> %ret
}

; Test v8i16.
define <8 x i16> @f2(i16 %val) {
; CHECK-LABEL: f2:
; CHECK: vlvgh %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <8 x i16> undef, i16 %val, i32 0
  ret <8 x i16> %ret
}

; Test v4i32.
define <4 x i32> @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: vlvgf %v24, %r2, 0
; CHECK: br %r14
  %ret = insertelement <4 x i32> undef, i32 %val, i32 0
  ret <4 x i32> %ret
}

; Test v2i64.  Here we load %val into both halves.
define <2 x i64> @f4(i64 %val) {
; CHECK-LABEL: f4:
; CHECK: vlvgp %v24, %r2, %r2
; CHECK: br %r14
  %ret = insertelement <2 x i64> undef, i64 %val, i32 0
  ret <2 x i64> %ret
}

; Test v4f32, which is just a move.
define <4 x float> @f5(float %val) {
; CHECK-LABEL: f5:
; CHECK: vlr %v24, %v0
; CHECK: br %r14
  %ret = insertelement <4 x float> undef, float %val, i32 0
  ret <4 x float> %ret
}

; Likewise v2f64.
define <2 x double> @f6(double %val) {
; CHECK-LABEL: f6:
; CHECK: vlr %v24, %v0
; CHECK: br %r14
  %ret = insertelement <2 x double> undef, double %val, i32 0
  ret <2 x double> %ret
}
