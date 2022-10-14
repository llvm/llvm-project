; Test 128-bit floating-point division on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

define void @f1(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfdxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %sum = fdiv fp128 %f1, %f2
  store fp128 %sum, ptr %ptr1
  ret void
}
