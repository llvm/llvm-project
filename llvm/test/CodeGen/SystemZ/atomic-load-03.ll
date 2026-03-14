; Test 32-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i32 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: l %r2, 0(%r2)
; CHECK: br %r14
  %val = load atomic i32, ptr %src seq_cst, align 4
  ret i32 %val
}
