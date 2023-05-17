; Test f128 floating-point strict truncations/extensions on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.fptrunc.f32.f128(fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f128(fp128, metadata, metadata)

declare fp128 @llvm.experimental.constrained.fpext.f128.f32(float, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

; Test f128->f64.
define double @f1(ptr %ptr) #0 {
; CHECK-LABEL: f1:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 0
; CHECK: br %r14
  %val = load fp128, ptr %ptr
  %res = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %res
}

; Test f128->f32.
define float @f2(ptr %ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wflrx %f0, [[REG]], 0, 3
; CHECK: ledbra %f0, 0, %f0, 0
; CHECK: br %r14
  %val = load fp128, ptr %ptr
  %res = call float @llvm.experimental.constrained.fptrunc.f32.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %res
}

; Test f64->f128.
define void @f3(ptr %dst, double %val) #0 {
; CHECK-LABEL: f3:
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Test f32->f128.
define void @f4(ptr %dst, float %val) #0 {
; CHECK-LABEL: f4:
; CHECK: ldebr %f0, %f0
; CHECK: wflld [[RES:%v[0-9]+]], %f0
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %res = call fp128 @llvm.experimental.constrained.fpext.f128.f32(float %val,
                                               metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

attributes #0 = { strictfp }
