; Test 32-bit addition in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check additions of 1.
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: ahi %r2, 1
; CHECK: br %r14
  %add = add i32 %a, 1
  ret i32 %add
}

; Check the high end of the AHI range.
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: ahi %r2, 32767
; CHECK: br %r14
  %add = add i32 %a, 32767
  ret i32 %add
}

; Check the next value up, which must use AFI instead.
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: afi %r2, 32768
; CHECK: br %r14
  %add = add i32 %a, 32768
  ret i32 %add
}

; Check the high end of the signed 32-bit range.
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: afi %r2, 2147483647
; CHECK: br %r14
  %add = add i32 %a, 2147483647
  ret i32 %add
}

; Check the next value up, which is treated as a negative value.
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: afi %r2, -2147483648
; CHECK: br %r14
  %add = add i32 %a, 2147483648
  ret i32 %add
}

; Check the high end of the negative AHI range.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: ahi %r2, -1
; CHECK: br %r14
  %add = add i32 %a, -1
  ret i32 %add
}

; Check the low end of the AHI range.
define i32 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: ahi %r2, -32768
; CHECK: br %r14
  %add = add i32 %a, -32768
  ret i32 %add
}

; Check the next value down, which must use AFI instead.
define i32 @f8(i32 %a) {
; CHECK-LABEL: f8:
; CHECK: afi %r2, -32769
; CHECK: br %r14
  %add = add i32 %a, -32769
  ret i32 %add
}

; Check the low end of the signed 32-bit range.
define i32 @f9(i32 %a) {
; CHECK-LABEL: f9:
; CHECK: afi %r2, -2147483648
; CHECK: br %r14
  %add = add i32 %a, -2147483648
  ret i32 %add
}

; Check the next value down, which is treated as a positive value.
define i32 @f10(i32 %a) {
; CHECK-LABEL: f10:
; CHECK: afi %r2, 2147483647
; CHECK: br %r14
  %add = add i32 %a, -2147483649
  ret i32 %add
}
