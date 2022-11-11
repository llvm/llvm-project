; Test 16-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i16 @f1(ptr %src) {
; CHECK-LABEL: f1:
; CHECK: lh %r2, 0(%r2)
; CHECK: br %r14
  %val = load atomic i16, ptr %src seq_cst, align 2
  ret i16 %val
}
