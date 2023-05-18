; Test the three-operand forms of OR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check XRK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: ork %r2, %r3, %r4
; CHECK: br %r14
  %or = or i32 %b, %c
  ret i32 %or
}

; Check that we can still use OR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: or %r2, %r3
; CHECK: br %r14
  %or = or i32 %a, %b
  ret i32 %or
}

; Check OGRK.
define i64 @f3(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f3:
; CHECK: ogrk %r2, %r3, %r4
; CHECK: br %r14
  %or = or i64 %b, %c
  ret i64 %or
}

; Check that we can still use OGR in obvious cases.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: ogr %r2, %r3
; CHECK: br %r14
  %or = or i64 %a, %b
  ret i64 %or
}
