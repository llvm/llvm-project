; Test rounding functions for z14 and above.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test rint for f32.
declare float @llvm.rint.f32(float %f)
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: fiebra %f0, 0, %f0, 0
; CHECK: br %r14
  %res = call float @llvm.rint.f32(float %f)
  ret float %res
}

; Test rint for f64.
declare double @llvm.rint.f64(double %f)
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: fidbra %f0, 0, %f0, 0
; CHECK: br %r14
  %res = call double @llvm.rint.f64(double %f)
  ret double %res
}

; Test rint for f128.
declare fp128 @llvm.rint.f128(fp128 %f)
define void @f3(fp128 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 0, 0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.rint.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test nearbyint for f32.
declare float @llvm.nearbyint.f32(float %f)
define float @f4(float %f) {
; CHECK-LABEL: f4:
; CHECK: fiebra %f0, 0, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.nearbyint.f32(float %f)
  ret float %res
}

; Test nearbyint for f64.
declare double @llvm.nearbyint.f64(double %f)
define double @f5(double %f) {
; CHECK-LABEL: f5:
; CHECK: fidbra %f0, 0, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.nearbyint.f64(double %f)
  ret double %res
}

; Test nearbyint for f128.
declare fp128 @llvm.nearbyint.f128(fp128 %f)
define void @f6(fp128 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.nearbyint.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test floor for f32.
declare float @llvm.floor.f32(float %f)
define float @f7(float %f) {
; CHECK-LABEL: f7:
; CHECK: fiebra %f0, 7, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.floor.f32(float %f)
  ret float %res
}

; Test floor for f64.
declare double @llvm.floor.f64(double %f)
define double @f8(double %f) {
; CHECK-LABEL: f8:
; CHECK: fidbra %f0, 7, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.floor.f64(double %f)
  ret double %res
}

; Test floor for f128.
declare fp128 @llvm.floor.f128(fp128 %f)
define void @f9(fp128 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 7
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.floor.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test ceil for f32.
declare float @llvm.ceil.f32(float %f)
define float @f10(float %f) {
; CHECK-LABEL: f10:
; CHECK: fiebra %f0, 6, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.ceil.f32(float %f)
  ret float %res
}

; Test ceil for f64.
declare double @llvm.ceil.f64(double %f)
define double @f11(double %f) {
; CHECK-LABEL: f11:
; CHECK: fidbra %f0, 6, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.ceil.f64(double %f)
  ret double %res
}

; Test ceil for f128.
declare fp128 @llvm.ceil.f128(fp128 %f)
define void @f12(fp128 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 6
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.ceil.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test trunc for f32.
declare float @llvm.trunc.f32(float %f)
define float @f13(float %f) {
; CHECK-LABEL: f13:
; CHECK: fiebra %f0, 5, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.trunc.f32(float %f)
  ret float %res
}

; Test trunc for f64.
declare double @llvm.trunc.f64(double %f)
define double @f14(double %f) {
; CHECK-LABEL: f14:
; CHECK: fidbra %f0, 5, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.trunc.f64(double %f)
  ret double %res
}

; Test trunc for f128.
declare fp128 @llvm.trunc.f128(fp128 %f)
define void @f15(fp128 *%ptr) {
; CHECK-LABEL: f15:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 5
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.trunc.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test round for f32.
declare float @llvm.round.f32(float %f)
define float @f16(float %f) {
; CHECK-LABEL: f16:
; CHECK: fiebra %f0, 1, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.round.f32(float %f)
  ret float %res
}

; Test round for f64.
declare double @llvm.round.f64(double %f)
define double @f17(double %f) {
; CHECK-LABEL: f17:
; CHECK: fidbra %f0, 1, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.round.f64(double %f)
  ret double %res
}

; Test round for f128.
declare fp128 @llvm.round.f128(fp128 %f)
define void @f18(fp128 *%ptr) {
; CHECK-LABEL: f18:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 1
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128 , fp128 *%ptr
  %res = call fp128 @llvm.round.f128(fp128 %src)
  store fp128 %res, fp128 *%ptr
  ret void
}
