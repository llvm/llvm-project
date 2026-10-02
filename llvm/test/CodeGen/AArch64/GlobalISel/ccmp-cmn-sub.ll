; NOTE: Assertions may be updated by utils/update_llc_test_checks.py
; RUN: llc -mtriple=aarch64-- -global-isel -verify-machineinstrs %s -o - | FileCheck %s

define i32 @ccmp_cmn_rhs(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) {
; CHECK-LABEL: ccmp_cmn_rhs:
; CHECK-DAG: cmp w0, #5
; CHECK: ccmn w2, w1, #0, eq
; CHECK-NEXT: csel
entry:
  %neg = sub nsw i32 0, %b
  %cmp_sub = icmp eq i32 %c, %neg
  %cmp_simple = icmp eq i32 %a, 5
  %and = and i1 %cmp_sub, %cmp_simple
  %s = select i1 %and, i32 %x, i32 %y
  ret i32 %s
}

; LHS of icmp is (sub 0, %a) with equality: commute to CCMN.

define i32 @ccmp_cmn_lhs_eq(i32 %a, i32 %b, i32 %c, i32 %x, i32 %y) {
; CHECK-LABEL: ccmp_cmn_lhs_eq:
; CHECK-DAG: cmp w2, #7
; CHECK: ccmn w0, w1, #0, eq
; CHECK-NEXT: csel
entry:
  %neg = sub nsw i32 0, %a
  %cmp_sub = icmp eq i32 %neg, %b
  %cmp_simple = icmp eq i32 %c, 7
  %and = and i1 %cmp_sub, %cmp_simple
  %s = select i1 %and, i32 %x, i32 %y
  ret i32 %s
}
