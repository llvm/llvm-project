; Test the three-operand forms of subtraction.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check SRK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: srk %r2, %r3, %r4
; CHECK: br %r14
  %sub = sub i32 %b, %c
  ret i32 %sub
}

; Check that we can still use SR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: sr %r2, %r3
; CHECK: br %r14
  %sub = sub i32 %a, %b
  ret i32 %sub
}

; Check SGRK.
define i64 @f3(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f3:
; CHECK: sgrk %r2, %r3, %r4
; CHECK: br %r14
  %sub = sub i64 %b, %c
  ret i64 %sub
}

; Check that we can still use SGR in obvious cases.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: sgr %r2, %r3
; CHECK: br %r14
  %sub = sub i64 %a, %b
  ret i64 %sub
}
