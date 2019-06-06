; Test strict vector minimum on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.experimental.constrained.minnum.f64(double, double, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.minnum.v2f64(<2 x double>, <2 x double>, metadata, metadata)

declare float @llvm.experimental.constrained.minnum.f32(float, float, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.minnum.v4f32(<4 x float>, <4 x float>, metadata, metadata)

declare fp128 @llvm.experimental.constrained.minnum.f128(fp128, fp128, metadata, metadata)

; Test the f64 minnum intrinsic.
define double @f1(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f1:
; CHECK: wfmindb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @llvm.experimental.constrained.minnum.f64(
                        double %val1, double %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %ret
}

; Test the v2f64 minnum intrinsic.
define <2 x double> @f2(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f2:
; CHECK: vfmindb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.minnum.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <2 x double> %ret
}

; Test the f32 minnum intrinsic.
define float @f3(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f3:
; CHECK: wfminsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @llvm.experimental.constrained.minnum.f32(
                        float %val1, float %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %ret
}

; Test the v4f32 minnum intrinsic.
define <4 x float> @f4(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) {
; CHECK-LABEL: f4:
; CHECK: vfminsb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.minnum.v4f32(
                        <4 x float> %val1, <4 x float> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %ret
}

; Test the f128 minnum intrinsic.
define void @f5(fp128 *%ptr1, fp128 *%ptr2, fp128 *%dst) {
; CHECK-LABEL: f5:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfminxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 4
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, fp128* %ptr1
  %val2 = load fp128, fp128* %ptr2
  %res = call fp128 @llvm.experimental.constrained.minnum.f128(
                        fp128 %val1, fp128 %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128* %dst
  ret void
}

