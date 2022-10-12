; Test 64-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i64 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r2)
; CHECK: br %r14
  %val = load atomic i64, ptr %src seq_cst, align 8
  ret i64 %val
}
