; Test 8-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i8 %val, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: stc %r2, 0(%r3)
; CHECK: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i8 %val, ptr %src seq_cst, align 1
  ret void
}

define void @f2(i8 %val, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: stc %r2, 0(%r3)
; CHECK-NOT: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i8 %val, ptr %src monotonic, align 1
  ret void
}
