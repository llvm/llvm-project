; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.fma.f128(fp128 %f1, fp128 %f2, fp128 %f3)

define void @f1(ptr %ptr1, ptr %ptr2, ptr %ptr3, ptr %dst) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: vl [[REG3:%v[0-9]+]], 0(%r4)
; CHECK: wfmaxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], [[REG3]]
; CHECK: vst [[RES]], 0(%r5)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %f3 = load fp128, ptr %ptr3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %f3)
  store fp128 %res, ptr %dst
  ret void
}

define void @f2(ptr %ptr1, ptr %ptr2, ptr %ptr3, ptr %dst) {
; CHECK-LABEL: f2:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: vl [[REG3:%v[0-9]+]], 0(%r4)
; CHECK: wfmsxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], [[REG3]]
; CHECK: vst [[RES]], 0(%r5)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %f3 = load fp128, ptr %ptr3
  %neg = fsub fp128 0xL00000000000000008000000000000000, %f3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %neg)
  store fp128 %res, ptr %dst
  ret void
}

define void @f3(ptr %ptr1, ptr %ptr2, ptr %ptr3, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: vl [[REG3:%v[0-9]+]], 0(%r4)
; CHECK: wfnmaxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], [[REG3]]
; CHECK: vst [[RES]], 0(%r5)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %f3 = load fp128, ptr %ptr3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %f3)
  %negres = fsub fp128 0xL00000000000000008000000000000000, %res
  store fp128 %negres, ptr %dst
  ret void
}

define void @f4(ptr %ptr1, ptr %ptr2, ptr %ptr3, ptr %dst) {
; CHECK-LABEL: f4:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK-DAG: vl [[REG3:%v[0-9]+]], 0(%r4)
; CHECK: wfnmsxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], [[REG3]]
; CHECK: vst [[RES]], 0(%r5)
; CHECK: br %r14
  %f1 = load fp128, ptr %ptr1
  %f2 = load fp128, ptr %ptr2
  %f3 = load fp128, ptr %ptr3
  %neg = fsub fp128 0xL00000000000000008000000000000000, %f3
  %res = call fp128 @llvm.fma.f128 (fp128 %f1, fp128 %f2, fp128 %neg)
  %negres = fsub fp128 0xL00000000000000008000000000000000, %res
  store fp128 %negres, ptr %dst
  ret void
}

