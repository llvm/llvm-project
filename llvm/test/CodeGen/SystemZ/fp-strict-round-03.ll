; Test strict rounding functions for z14 and above.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test rint for f32.
declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: fiebra %f0, 0, %f0, 0
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.rint.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test rint for f64.
declare double @llvm.experimental.constrained.rint.f64(double, metadata, metadata)
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: fidbra %f0, 0, %f0, 0
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.rint.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test rint for f128.
declare fp128 @llvm.experimental.constrained.rint.f128(fp128, metadata, metadata)
define void @f3(fp128 *%ptr) {
; CHECK-LABEL: f3:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 0, 0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.rint.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test nearbyint for f32.
declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
define float @f4(float %f) {
; CHECK-LABEL: f4:
; CHECK: fiebra %f0, 0, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.nearbyint.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test nearbyint for f64.
declare double @llvm.experimental.constrained.nearbyint.f64(double, metadata, metadata)
define double @f5(double %f) {
; CHECK-LABEL: f5:
; CHECK: fidbra %f0, 0, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.nearbyint.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test nearbyint for f128.
declare fp128 @llvm.experimental.constrained.nearbyint.f128(fp128, metadata, metadata)
define void @f6(fp128 *%ptr) {
; CHECK-LABEL: f6:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.nearbyint.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test floor for f32.
declare float @llvm.experimental.constrained.floor.f32(float, metadata, metadata)
define float @f7(float %f) {
; CHECK-LABEL: f7:
; CHECK: fiebra %f0, 7, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.floor.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test floor for f64.
declare double @llvm.experimental.constrained.floor.f64(double, metadata, metadata)
define double @f8(double %f) {
; CHECK-LABEL: f8:
; CHECK: fidbra %f0, 7, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.floor.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test floor for f128.
declare fp128 @llvm.experimental.constrained.floor.f128(fp128, metadata, metadata)
define void @f9(fp128 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 7
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.floor.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test ceil for f32.
declare float @llvm.experimental.constrained.ceil.f32(float, metadata, metadata)
define float @f10(float %f) {
; CHECK-LABEL: f10:
; CHECK: fiebra %f0, 6, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.ceil.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test ceil for f64.
declare double @llvm.experimental.constrained.ceil.f64(double, metadata, metadata)
define double @f11(double %f) {
; CHECK-LABEL: f11:
; CHECK: fidbra %f0, 6, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.ceil.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test ceil for f128.
declare fp128 @llvm.experimental.constrained.ceil.f128(fp128, metadata, metadata)
define void @f12(fp128 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 6
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.ceil.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test trunc for f32.
declare float @llvm.experimental.constrained.trunc.f32(float, metadata, metadata)
define float @f13(float %f) {
; CHECK-LABEL: f13:
; CHECK: fiebra %f0, 5, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.trunc.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test trunc for f64.
declare double @llvm.experimental.constrained.trunc.f64(double, metadata, metadata)
define double @f14(double %f) {
; CHECK-LABEL: f14:
; CHECK: fidbra %f0, 5, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.trunc.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test trunc for f128.
declare fp128 @llvm.experimental.constrained.trunc.f128(fp128, metadata, metadata)
define void @f15(fp128 *%ptr) {
; CHECK-LABEL: f15:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 5
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.trunc.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

; Test round for f32.
declare float @llvm.experimental.constrained.round.f32(float, metadata, metadata)
define float @f16(float %f) {
; CHECK-LABEL: f16:
; CHECK: fiebra %f0, 1, %f0, 4
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.round.f32(
                        float %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

; Test round for f64.
declare double @llvm.experimental.constrained.round.f64(double, metadata, metadata)
define double @f17(double %f) {
; CHECK-LABEL: f17:
; CHECK: fidbra %f0, 1, %f0, 4
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.round.f64(
                        double %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Test round for f128.
declare fp128 @llvm.experimental.constrained.round.f128(fp128, metadata, metadata)
define void @f18(fp128 *%ptr) {
; CHECK-LABEL: f18:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfixb [[RES:%v[0-9]+]], [[REG]], 4, 1
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %src = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.round.f128(
                        fp128 %src,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}

