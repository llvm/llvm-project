; Test the three-operand forms of addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check ARK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: ark %r2, %r3, %r4
; CHECK: br %r14
  %add = add i32 %b, %c
  ret i32 %add
}

; Check that we can still use AR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: ar %r2, %r3
; CHECK: br %r14
  %add = add i32 %a, %b
  ret i32 %add
}

; Check AGRK.
define i64 @f3(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f3:
; CHECK: agrk %r2, %r3, %r4
; CHECK: br %r14
  %add = add i64 %b, %c
  ret i64 %add
}

; Check that we can still use AGR in obvious cases.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: agr %r2, %r3
; CHECK: br %r14
  %add = add i64 %a, %b
  ret i64 %add
}
