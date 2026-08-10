; Testing peephole for generating shorter code for (and (xor b, -1), a)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: ngr %r3, %r2
; CHECK: xgr %r2, %r3
; CHECK: br %r14
  %neg = xor i64 %b, -1
  %and = and i64 %neg, %a
  ret i64 %and
}

