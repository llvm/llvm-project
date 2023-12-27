; Test double atomic stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %dst, double %val) {
; CHECK-LABEL: f1:
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  store atomic double %val, ptr %dst seq_cst, align 8
  ret void
}
