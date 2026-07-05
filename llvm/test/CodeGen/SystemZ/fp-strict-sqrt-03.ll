; Test strict 128-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.sqrt.f128(fp128, metadata, metadata)

; There's no memory form of SQXBR.
define void @f1(ptr %ptr) strictfp {
; CHECK-LABEL: f1:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: sqxbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %orig = load fp128, ptr %ptr
  %sqrt = call fp128 @llvm.experimental.constrained.sqrt.f128(
                        fp128 %orig,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") strictfp
  store fp128 %sqrt, ptr %ptr
  ret void
}
