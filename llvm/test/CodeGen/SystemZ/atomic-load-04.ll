; Test 64-bit atomic loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i64 @f1(i64 *%src) {
; CHECK-LABEL: f1:
; CHECK: lg %r2, 0(%r2)
; CHECK: br %r14
  %val = load atomic i64, i64 *%src seq_cst, align 8
  ret i64 %val
}
