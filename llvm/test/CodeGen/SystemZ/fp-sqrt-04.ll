; Test 128-bit floating-point square root on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.sqrt.f128(fp128 %f)

define void @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfsqxb [[RES:%v[0-9]+]], [[REG]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f = load fp128, ptr %ptr
  %res = call fp128 @llvm.sqrt.f128(fp128 %f)
  store fp128 %res, ptr %ptr
  ret void
}
