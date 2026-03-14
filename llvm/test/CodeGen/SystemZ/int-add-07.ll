; Test 64-bit addition in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check additions of 1.
define i64 @f1(i64 %a) {
; CHECK-LABEL: f1:
; CHECK: {{aghi %r2, 1|la %r[0-5], 1\(%r2\)}}
; CHECK: br %r14
  %add = add i64 %a, 1
  ret i64 %add
}

; Check the high end of the AGHI range.
define i64 @f2(i64 %a) {
; CHECK-LABEL: f2:
; CHECK: aghi %r2, 32767
; CHECK: br %r14
  %add = add i64 %a, 32767
  ret i64 %add
}

; Check the next value up, which must use AGFI instead.
define i64 @f3(i64 %a) {
; CHECK-LABEL: f3:
; CHECK: {{agfi %r2, 32768|lay %r[0-5], 32768\(%r2\)}}
; CHECK: br %r14
  %add = add i64 %a, 32768
  ret i64 %add
}

; Check the high end of the AGFI range.
define i64 @f4(i64 %a) {
; CHECK-LABEL: f4:
; CHECK: agfi %r2, 2147483647
; CHECK: br %r14
  %add = add i64 %a, 2147483647
  ret i64 %add
}

; Check the next value up, which must use ALGFI instead.
define i64 @f5(i64 %a) {
; CHECK-LABEL: f5:
; CHECK: algfi %r2, 2147483648
; CHECK: br %r14
  %add = add i64 %a, 2147483648
  ret i64 %add
}

; Check the high end of the ALGFI range.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: algfi %r2, 4294967295
; CHECK: br %r14
  %add = add i64 %a, 4294967295
  ret i64 %add
}

; Check the next value up, which must be loaded into a register first.
define i64 @f7(i64 %a) {
; CHECK-LABEL: f7:
; CHECK: llihl %r0, 1
; CHECK: agr
; CHECK: br %r14
  %add = add i64 %a, 4294967296
  ret i64 %add
}

; Check the high end of the negative AGHI range.
define i64 @f8(i64 %a) {
; CHECK-LABEL: f8:
; CHECK: aghi %r2, -1
; CHECK: br %r14
  %add = add i64 %a, -1
  ret i64 %add
}

; Check the low end of the AGHI range.
define i64 @f9(i64 %a) {
; CHECK-LABEL: f9:
; CHECK: aghi %r2, -32768
; CHECK: br %r14
  %add = add i64 %a, -32768
  ret i64 %add
}

; Check the next value down, which must use AGFI instead.
define i64 @f10(i64 %a) {
; CHECK-LABEL: f10:
; CHECK: {{agfi %r2, -32769|lay %r[0-5]+, -32769\(%r2\)}}
; CHECK: br %r14
  %add = add i64 %a, -32769
  ret i64 %add
}

; Check the low end of the AGFI range.
define i64 @f11(i64 %a) {
; CHECK-LABEL: f11:
; CHECK: agfi %r2, -2147483648
; CHECK: br %r14
  %add = add i64 %a, -2147483648
  ret i64 %add
}

; Check the next value down, which must use SLGFI instead.
define i64 @f12(i64 %a) {
; CHECK-LABEL: f12:
; CHECK: slgfi %r2, 2147483649
; CHECK: br %r14
  %add = add i64 %a, -2147483649
  ret i64 %add
}

; Check the low end of the SLGFI range.
define i64 @f13(i64 %a) {
; CHECK-LABEL: f13:
; CHECK: slgfi %r2, 4294967295
; CHECK: br %r14
  %add = add i64 %a, -4294967295
  ret i64 %add
}

; Check the next value down, which must use register addition instead.
define i64 @f14(i64 %a) {
; CHECK-LABEL: f14:
; CHECK: llihf %r0, 4294967295
; CHECK: agr
; CHECK: br %r14
  %add = add i64 %a, -4294967296
  ret i64 %add
}
