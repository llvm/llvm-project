; Test three-operand multiplication instructions on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

; Check MSRKC.
define i32 @f1(i32 %dummy, i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: msrkc %r2, %r3, %r4
; CHECK: br %r14
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check MSGRKC.
define i64 @f2(i64 %dummy, i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: msgrkc %r2, %r3, %r4
; CHECK: br %r14
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Verify that we still use MSGFR for i32->i64 multiplies.
define i64 @f3(i64 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: msgfr %r2, %r3
; CHECK: br %r14
  %bext = sext i32 %b to i64
  %mul = mul i64 %a, %bext
  ret i64 %mul
}

