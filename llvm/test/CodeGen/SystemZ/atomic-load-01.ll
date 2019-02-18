; Test 8-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i8 @f1(i8 *%src) {
; CHECK-LABEL: f1:
; CHECK: lb %r2, 0(%r2)
; CHECK: br %r14
  %val = load atomic i8, i8 *%src seq_cst, align 1
  ret i8 %val
}
