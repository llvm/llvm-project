; Test insertions of register values into 0.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v16i8 insertion into 0.
define <16 x i8> @f1(i8 %val1, i8 %val2) {
; CHECK-LABEL: f1:
; CHECK: vgbm %v24, 0
; CHECK-DAG: vlvgb %v24, %r2, 2
; CHECK-DAG: vlvgb %v24, %r3, 12
; CHECK: br %r14
  %vec1 = insertelement <16 x i8> zeroinitializer, i8 %val1, i32 2
  %vec2 = insertelement <16 x i8> %vec1, i8 %val2, i32 12
  ret <16 x i8> %vec2
}

; Test v8i16 insertion into 0.
define <8 x i16> @f2(i16 %val1, i16 %val2) {
; CHECK-LABEL: f2:
; CHECK: vgbm %v24, 0
; CHECK-DAG: vlvgh %v24, %r2, 3
; CHECK-DAG: vlvgh %v24, %r3, 5
; CHECK: br %r14
  %vec1 = insertelement <8 x i16> zeroinitializer, i16 %val1, i32 3
  %vec2 = insertelement <8 x i16> %vec1, i16 %val2, i32 5
  ret <8 x i16> %vec2
}

; Test v4i32 insertion into 0.
define <4 x i32> @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: vgbm %v24, 0
; CHECK: vlvgf %v24, %r2, 3
; CHECK: br %r14
  %ret = insertelement <4 x i32> zeroinitializer, i32 %val, i32 3
  ret <4 x i32> %ret
}

; Test v2i64 insertion into 0.
define <2 x i64> @f4(i64 %val) {
; CHECK-LABEL: f4:
; CHECK: lghi [[REG:%r[0-5]]], 0
; CHECK: vlvgp %v24, [[REG]], %r2
; CHECK: br %r14
  %ret = insertelement <2 x i64> zeroinitializer, i64 %val, i32 1
  ret <2 x i64> %ret
}

; Test v4f32 insertion into 0.
define <4 x float> @f5(float %val) {
; CHECK-LABEL: f5:
; CHECK-DAG: vuplhf [[REG:%v[0-9]+]], %v0
; CHECK-DAG: vgbm [[ZERO:%v[0-9]+]], 0
; CHECK: vmrhg %v24, [[ZERO]], [[REG]]
; CHECK: br %r14
  %ret = insertelement <4 x float> zeroinitializer, float %val, i32 3
  ret <4 x float> %ret
}

; Test v2f64 insertion into 0.
define <2 x double> @f6(double %val) {
; CHECK-LABEL: f6:
; CHECK: vgbm [[REG:%v[0-9]+]], 0
; CHECK: vmrhg %v24, [[REG]], %v0
; CHECK: br %r14
  %ret = insertelement <2 x double> zeroinitializer, double %val, i32 1
  ret <2 x double> %ret
}

