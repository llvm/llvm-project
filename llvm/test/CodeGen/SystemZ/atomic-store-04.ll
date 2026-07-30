; Test 64-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i64 %val, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: stg %r2, 0(%r3)
; CHECK: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i64 %val, ptr %src seq_cst, align 8
  ret void
}

define void @f2(i64 %val, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: stg %r2, 0(%r3)
; CHECK-NOT: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i64 %val, ptr %src monotonic, align 8
  ret void
}
