; Test 32-bit atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(i32 %val, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: st %r2, 0(%r3)
; CHECK: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i32 %val, ptr %src seq_cst, align 4
  ret void
}

define void @f2(i32 %val, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: st %r2, 0(%r3)
; CHECK-NOT: bcr 1{{[45]}}, %r0
; CHECK: br %r14
  store atomic i32 %val, ptr %src monotonic, align 4
  ret void
}
