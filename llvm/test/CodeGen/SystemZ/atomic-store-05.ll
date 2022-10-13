; Test 128-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i128 %val, ptr %src) {
; CHECK-LABEL: f1:
; CHECK-DAG: lg %r1, 8(%r2)
; CHECK-DAG: lg %r0, 0(%r2)
; CHECK: stpq %r0, 0(%r3)
; CHECK: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i128 %val, ptr %src seq_cst, align 16
  ret void
}

define void @f2(i128 %val, ptr %src) {
; CHECK-LABEL: f2:
; CHECK-DAG: lg %r1, 8(%r2)
; CHECK-DAG: lg %r0, 0(%r2)
; CHECK: stpq %r0, 0(%r3)
; CHECK-NOT: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i128 %val, ptr %src monotonic, align 16
  ret void
}
