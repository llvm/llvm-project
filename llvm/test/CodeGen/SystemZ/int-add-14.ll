; Test 32-bit addition in which the second operand is constant and in which
; three-operand forms are available.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check additions of 1.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: ahik %r2, %r3, 1
; CHECK: br %r14
  %add = add i32 %b, 1
  ret i32 %add
}

; Check the high end of the AHIK range.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: ahik %r2, %r3, 32767
; CHECK: br %r14
  %add = add i32 %b, 32767
  ret i32 %add
}

; Check the next value up, which must use AFI instead.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: afi {{%r[0-5]}}, 32768
; CHECK: br %r14
  %add = add i32 %b, 32768
  ret i32 %add
}

; Check the high end of the negative AHIK range.
define i32 @f4(i32 %a, i32 %b) {
; CHECK-LABEL: f4:
; CHECK: ahik %r2, %r3, -1
; CHECK: br %r14
  %add = add i32 %b, -1
  ret i32 %add
}

; Check the low end of the AHIK range.
define i32 @f5(i32 %a, i32 %b) {
; CHECK-LABEL: f5:
; CHECK: ahik %r2, %r3, -32768
; CHECK: br %r14
  %add = add i32 %b, -32768
  ret i32 %add
}

; Check the next value down, which must use AFI instead.
define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: afi {{%r[0-5]}}, -32769
; CHECK: br %r14
  %add = add i32 %b, -32769
  ret i32 %add
}

; Check that AHI is still used in obvious cases.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: ahi %r2, 1
; CHECK: br %r14
  %add = add i32 %a, 1
  ret i32 %add
}
