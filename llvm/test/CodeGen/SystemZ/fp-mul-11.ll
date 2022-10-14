; Test 128-bit floating-point multiplication on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

define void @f1(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %sum = fmul fp128 %f1, %f2
  store fp128 %sum, ptr %ptr1
  ret void
}

define void @f2(double %f1, double %f2, ptr %dst) {
; CHECK-LABEL: f2:
; CHECK-DAG: wflld [[REG1:%v[0-9]+]], %f0
; CHECK-DAG: wflld [[REG2:%v[0-9]+]], %f2
; CHECK: wfmxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = fmul fp128 %f1x, %f2x
  store fp128 %res, ptr %dst
  ret void
}

