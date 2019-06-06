; Test strict 128-bit floating-point square root on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare fp128 @llvm.experimental.constrained.sqrt.f128(fp128, metadata, metadata)

define void @f1(fp128 *%ptr) {
; CHECK-LABEL: f1:
; CHECK-DAG: vl [[REG:%v[0-9]+]], 0(%r2)
; CHECK: wfsqxb [[RES:%v[0-9]+]], [[REG]]
; CHECK: vst [[RES]], 0(%r2)
; CHECK: br %r14
  %f = load fp128, fp128 *%ptr
  %res = call fp128 @llvm.experimental.constrained.sqrt.f128(
                        fp128 %f,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%ptr
  ret void
}
