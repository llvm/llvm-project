; Test the three-operand forms of AND.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check NRK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: nrk %r2, %r3, %r4
; CHECK: br %r14
  %and = and i32 %b, %c
  ret i32 %and
}

; Check that we can still use NR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: nr %r2, %r3
; CHECK: br %r14
  %and = and i32 %a, %b
  ret i32 %and
}

; Check NGRK.
define i64 @f3(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f3:
; CHECK: ngrk %r2, %r3, %r4
; CHECK: br %r14
  %and = and i64 %b, %c
  ret i64 %and
}

; Check that we can still use NGR in obvious cases.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: ngr %r2, %r3
; CHECK: br %r14
  %and = and i64 %a, %b
  ret i64 %and
}
