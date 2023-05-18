; Test 64-bit addition in which the second operand is constant and in which
; three-operand forms are available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check additions of 1.
define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: {{aghik %r2, %r3, 1|la %r2, 1\(%r3\)}}
; CHECK: br %r14
  %add = add i64 %b, 1
  ret i64 %add
}

; Check the high end of the AGHIK range.
define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: aghik %r2, %r3, 32767
; CHECK: br %r14
  %add = add i64 %b, 32767
  ret i64 %add
}

; Check the next value up, which must use AGFI instead.
define i64 @f3(i64 %a, i64 %b) {
; CHECK-LABEL: f3:
; CHECK: {{agfi %r[0-5], 32768|lay %r2, 32768\(%r3\)}}
; CHECK: br %r14
  %add = add i64 %b, 32768
  ret i64 %add
}

; Check the high end of the negative AGHIK range.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: aghik %r2, %r3, -1
; CHECK: br %r14
  %add = add i64 %b, -1
  ret i64 %add
}

; Check the low end of the AGHIK range.
define i64 @f5(i64 %a, i64 %b) {
; CHECK-LABEL: f5:
; CHECK: aghik %r2, %r3, -32768
; CHECK: br %r14
  %add = add i64 %b, -32768
  ret i64 %add
}

; Check the next value down, which must use AGFI instead.
define i64 @f6(i64 %a, i64 %b) {
; CHECK-LABEL: f6:
; CHECK: {{agfi %r[0-5], -32769|lay %r2, -32769\(%r3\)}}
; CHECK: br %r14
  %add = add i64 %b, -32769
  ret i64 %add
}

; Check that AGHI is still used in obvious cases.
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK: aghi %r2, 32000
; CHECK: br %r14
  %add = add i64 %a, 32000
  ret i64 %add
}
