; Test f128 floating-point strict truncations/extensions on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.fptrunc.f32.f128(fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f128(fp128, metadata, metadata)

declare fp128 @llvm.experimental.constrained.fpext.f128.f32(float, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

; Test f128->f64.
define double @f1(fp128 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 0
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %res = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %res
}

; Test f128->f32.
define float @f2(fp128 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 3
; CHECK: ledbra %f0, 0, %f0, 0
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %res = call float @llvm.experimental.constrained.fptrunc.f32.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret float %res
}

; Test f64->f128.
define void @f3(fp128 *%dst, double %val) {
; CHECK-LABEL: f3:
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Test f32->f128.
define void @f4(fp128 *%dst, float %val) {
; CHECK-LABEL: f4:
; CHECK: ldebr %f0, %f0
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f32(float %val,
                                               metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

