; Test 64-bit multiplication in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check multiplication by 2, which should use shifts.
define i64 @f1(i64 %a, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: sllg %r2, %r2, 1
; CHECK: br %r14
  %mul = mul i64 %a, 2
  ret i64 %mul
}

; Check multiplication by 3.
define i64 @f2(i64 %a, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK: mghi %r2, 3
; CHECK: br %r14
  %mul = mul i64 %a, 3
  ret i64 %mul
}

; Check the high end of the MGHI range.
define i64 @f3(i64 %a, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK: mghi %r2, 32767
; CHECK: br %r14
  %mul = mul i64 %a, 32767
  ret i64 %mul
}

; Check the next value up, which should use shifts.
define i64 @f4(i64 %a, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK: sllg %r2, %r2, 15
; CHECK: br %r14
  %mul = mul i64 %a, 32768
  ret i64 %mul
}

; Check the next value up again, which can use MSGFI.
define i64 @f5(i64 %a, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK: msgfi %r2, 32769
; CHECK: br %r14
  %mul = mul i64 %a, 32769
  ret i64 %mul
}

; Check the high end of the MSGFI range.
define i64 @f6(i64 %a, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK: msgfi %r2, 2147483647
; CHECK: br %r14
  %mul = mul i64 %a, 2147483647
  ret i64 %mul
}

; Check the next value up, which should use shifts.
define i64 @f7(i64 %a, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK: sllg %r2, %r2, 31
; CHECK: br %r14
  %mul = mul i64 %a, 2147483648
  ret i64 %mul
}

; Check the next value up again, which cannot use a constant multiplicatoin.
define i64 @f8(i64 %a, ptr %dest) {
; CHECK-LABEL: f8:
; CHECK-NOT: msgfi
; CHECK: br %r14
  %mul = mul i64 %a, 2147483649
  ret i64 %mul
}

; Check multiplication by -1, which is a negation.
define i64 @f9(i64 %a, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK: lcgr {{%r[0-5]}}, %r2
; CHECK: br %r14
  %mul = mul i64 %a, -1
  ret i64 %mul
}

; Check multiplication by -2, which should use shifts.
define i64 @f10(i64 %a, ptr %dest) {
; CHECK-LABEL: f10:
; CHECK: sllg [[SHIFTED:%r[0-5]]], %r2, 1
; CHECK: lcgr %r2, [[SHIFTED]]
; CHECK: br %r14
  %mul = mul i64 %a, -2
  ret i64 %mul
}

; Check multiplication by -3.
define i64 @f11(i64 %a, ptr %dest) {
; CHECK-LABEL: f11:
; CHECK: mghi %r2, -3
; CHECK: br %r14
  %mul = mul i64 %a, -3
  ret i64 %mul
}

; Check the lowest useful MGHI value.
define i64 @f12(i64 %a, ptr %dest) {
; CHECK-LABEL: f12:
; CHECK: mghi %r2, -32767
; CHECK: br %r14
  %mul = mul i64 %a, -32767
  ret i64 %mul
}

; Check the next value down, which should use shifts.
define i64 @f13(i64 %a, ptr %dest) {
; CHECK-LABEL: f13:
; CHECK: sllg [[SHIFTED:%r[0-5]]], %r2, 15
; CHECK: lcgr %r2, [[SHIFTED]]
; CHECK: br %r14
  %mul = mul i64 %a, -32768
  ret i64 %mul
}

; Check the next value down again, which can use MSGFI.
define i64 @f14(i64 %a, ptr %dest) {
; CHECK-LABEL: f14:
; CHECK: msgfi %r2, -32769
; CHECK: br %r14
  %mul = mul i64 %a, -32769
  ret i64 %mul
}

; Check the lowest useful MSGFI value.
define i64 @f15(i64 %a, ptr %dest) {
; CHECK-LABEL: f15:
; CHECK: msgfi %r2, -2147483647
; CHECK: br %r14
  %mul = mul i64 %a, -2147483647
  ret i64 %mul
}

; Check the next value down, which should use shifts.
define i64 @f16(i64 %a, ptr %dest) {
; CHECK-LABEL: f16:
; CHECK: sllg [[SHIFTED:%r[0-5]]], %r2, 31
; CHECK: lcgr %r2, [[SHIFTED]]
; CHECK: br %r14
  %mul = mul i64 %a, -2147483648
  ret i64 %mul
}

; Check the next value down again, which cannot use constant multiplication
define i64 @f17(i64 %a, ptr %dest) {
; CHECK-LABEL: f17:
; CHECK-NOT: msgfi
; CHECK: br %r14
  %mul = mul i64 %a, -2147483649
  ret i64 %mul
}
