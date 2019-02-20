; Test insertions of memory values into 0.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test VLLEZB.
define <16 x i8> @f1(i8 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vllezb %v24, 0(%r2)
; CHECK: br %r14
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> zeroinitializer, i8 %val, i32 7
  ret <16 x i8> %ret
}

; Test VLLEZB with the highest in-range offset.
define <16 x i8> @f2(i8 *%base) {
; CHECK-LABEL: f2:
; CHECK: vllezb %v24, 4095(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4095
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> zeroinitializer, i8 %val, i32 7
  ret <16 x i8> %ret
}

; Test VLLEZB with the next highest offset.
define <16 x i8> @f3(i8 *%base) {
; CHECK-LABEL: f3:
; CHECK-NOT: vllezb %v24, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 4096
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> zeroinitializer, i8 %val, i32 7
  ret <16 x i8> %ret
}

; Test that VLLEZB allows an index.
define <16 x i8> @f4(i8 *%base, i64 %index) {
; CHECK-LABEL: f4:
; CHECK: vllezb %v24, 0({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %ptr = getelementptr i8, i8 *%base, i64 %index
  %val = load i8, i8 *%ptr
  %ret = insertelement <16 x i8> zeroinitializer, i8 %val, i32 7
  ret <16 x i8> %ret
}

; Test VLLEZH.
define <8 x i16> @f5(i16 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: vllezh %v24, 0(%r2)
; CHECK: br %r14
  %val = load i16, i16 *%ptr
  %ret = insertelement <8 x i16> zeroinitializer, i16 %val, i32 3
  ret <8 x i16> %ret
}

; Test VLLEZF.
define <4 x i32> @f6(i32 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vllezf %v24, 0(%r2)
; CHECK: br %r14
  %val = load i32, i32 *%ptr
  %ret = insertelement <4 x i32> zeroinitializer, i32 %val, i32 1
  ret <4 x i32> %ret
}

; Test VLLEZG.
define <2 x i64> @f7(i64 *%ptr) {
; CHECK-LABEL: f7:
; CHECK: vllezg %v24, 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%ptr
  %ret = insertelement <2 x i64> zeroinitializer, i64 %val, i32 0
  ret <2 x i64> %ret
}

; Test VLLEZF with a float.
define <4 x float> @f8(float *%ptr) {
; CHECK-LABEL: f8:
; CHECK: vllezf %v24, 0(%r2)
; CHECK: br %r14
  %val = load float, float *%ptr
  %ret = insertelement <4 x float> zeroinitializer, float %val, i32 1
  ret <4 x float> %ret
}

; Test VLLEZG with a double.
define <2 x double> @f9(double *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vllezg %v24, 0(%r2)
; CHECK: br %r14
  %val = load double, double *%ptr
  %ret = insertelement <2 x double> zeroinitializer, double %val, i32 0
  ret <2 x double> %ret
}

; Test VLLEZF with a float when the result is stored to memory.
define void @f10(float *%ptr, <4 x float> *%res) {
; CHECK-LABEL: f10:
; CHECK: vllezf [[REG:%v[0-9]+]], 0(%r2)
; CHECK: vst [[REG]], 0(%r3)
; CHECK: br %r14
  %val = load float, float *%ptr
  %ret = insertelement <4 x float> zeroinitializer, float %val, i32 1
  store <4 x float> %ret, <4 x float> *%res
  ret void
}

; Test VLLEZG with a double when the result is stored to memory.
define void @f11(double *%ptr, <2 x double> *%res) {
; CHECK-LABEL: f11:
; CHECK: vllezg [[REG:%v[0-9]+]], 0(%r2)
; CHECK: vst [[REG]], 0(%r3)
; CHECK: br %r14
  %val = load double, double *%ptr
  %ret = insertelement <2 x double> zeroinitializer, double %val, i32 0
  store <2 x double> %ret, <2 x double> *%res
  ret void
}

; Test VLLEZG when the zeroinitializer is shared.
define void @f12(i64 *%ptr, <2 x i64> *%res) {
; CHECK-LABEL: f12:
; CHECK: vllezg [[REG:%v[0-9]+]], 0(%r2)
; CHECK: vst [[REG]], 0(%r3)
; CHECK: vllezg [[REG1:%v[0-9]+]], 0(%r2)
; CHECK: vst [[REG1]], 0(%r3)
; CHECK: br %r14
  %val = load volatile i64, i64 *%ptr
  %ret = insertelement <2 x i64> zeroinitializer, i64 %val, i32 0
  store volatile <2 x i64> %ret, <2 x i64> *%res
  %val1 = load volatile i64, i64 *%ptr
  %ret1 = insertelement <2 x i64> zeroinitializer, i64 %val1, i32 0
  store volatile <2 x i64> %ret1, <2 x i64> *%res
  ret void
}

