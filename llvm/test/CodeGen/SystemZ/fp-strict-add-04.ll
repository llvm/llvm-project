; Test strict 128-bit floating-point addition on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.experimental.constrained.fadd.f128(fp128, fp128, metadata, metadata)

define void @f1(fp128 *%ptr1, fp128 *%ptr2) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfaxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr1
  %f2 = load fp128, fp128 *%ptr2
  %sum = call fp128 @llvm.experimental.constrained.fadd.f128(
                        fp128 %f1, fp128 %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %sum, fp128 *%ptr1
  ret void
}
