; Test strict 128-bit floating-point subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fsub.f128(fp128, fp128, metadata, metadata)

; There is no memory form of 128-bit subtraction.
define void @f1(fp128 *%ptr, float %f2) {
; CHECK-LABEL: f1:
; CHECK-DAG: lxebr %f0, %f0
; CHECK-DAG: ld %f1, 0(%r2)
; CHECK-DAG: ld %f3, 8(%r2)
; CHECK: sxbr %f1, %f0
; CHECK: std %f1, 0(%r2)
; CHECK: std %f3, 8(%r2)
; CHECK: br %r14
  %f1 = load fp128, fp128 *%ptr
  %f2x = fpext float %f2 to fp128
  %sum = call fp128 @llvm.experimental.constrained.fsub.f128(
                        fp128 %f1, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %sum, fp128 *%ptr
  ret void
}
