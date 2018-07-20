; Test 128-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.sqrt.f128(fp128 %f)

; There's no memory form of SQXBR.
define void @f1(fp128 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: sqxbr %f0, %f0
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %orig = load fp128, fp128 *%ptr
  %sqrt = call fp128 @llvm.sqrt.f128(fp128 %orig)
  store fp128 %sqrt, fp128 *%ptr
  ret void
}
