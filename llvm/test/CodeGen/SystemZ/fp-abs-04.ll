; Test negated floating-point absolute on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Test f32.
declare float @llvm.fabs.f32(float %f)
define float @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: lndfr %f0, %f0
; CHECK: br %r14
  %abs = call float @llvm.fabs.f32(float %f)
  %res = fsub float -0.0, %abs
  ret float %res
}

; Test f64.
declare double @llvm.fabs.f64(double %f)
define double @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: lndfr %f0, %f0
; CHECK: br %r14
  %abs = call double @llvm.fabs.f64(double %f)
  %res = fsub double -0.0, %abs
  ret double %res
}

; Test f128.  With the loads and stores, a pure negative-absolute would
; probably be better implemented using an OI on the upper byte.  Do some
; extra processing so that using FPRs is unequivocally better.
declare fp128 @llvm.fabs.f128(fp128 %f)
define void @f3(fp128 *%ptr, fp128 *%ptr2) {
; CHECK-LABEL: f3:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: wflnxb [[NEGREG1:%v[0-9]+]], [[REG1]]
; CHECK: wfdxb [[RES:%v[0-9]+]], [[NEGREG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %orig = load fp128, fp128 *%ptr
  %abs = call fp128 @llvm.fabs.f128(fp128 %orig)
  %negabs = fsub fp128 0xL00000000000000008000000000000000, %abs
  %op2 = load fp128, fp128 *%ptr2
  %res = fdiv fp128 %negabs, %op2
  store fp128 %res, fp128 *%ptr
  ret void
}
