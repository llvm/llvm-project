; Test 32-bit multiplication in which the second operand is constant.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check multiplication by 2, which should use shifts.
define i32 @f1(i32 %a, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: sll %r2, 1
; CHECK: br %r14
  %mul = mul i32 %a, 2
  ret i32 %mul
}

; Check multiplication by 3.
define i32 @f2(i32 %a, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK: mhi %r2, 3
; CHECK: br %r14
  %mul = mul i32 %a, 3
  ret i32 %mul
}

; Check the high end of the MHI range.
define i32 @f3(i32 %a, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK: mhi %r2, 32767
; CHECK: br %r14
  %mul = mul i32 %a, 32767
  ret i32 %mul
}

; Check the next value up, which should use shifts.
define i32 @f4(i32 %a, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK: sll %r2, 15
; CHECK: br %r14
  %mul = mul i32 %a, 32768
  ret i32 %mul
}

; Check the next value up again, which can use MSFI.
define i32 @f5(i32 %a, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK: msfi %r2, 32769
; CHECK: br %r14
  %mul = mul i32 %a, 32769
  ret i32 %mul
}

; Check the high end of the MSFI range.
define i32 @f6(i32 %a, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK: msfi %r2, 2147483647
; CHECK: br %r14
  %mul = mul i32 %a, 2147483647
  ret i32 %mul
}

; Check the next value up, which should use shifts.
define i32 @f7(i32 %a, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK: sll %r2, 31
; CHECK: br %r14
  %mul = mul i32 %a, 2147483648
  ret i32 %mul
}

; Check the next value up again, which is treated as a negative value.
define i32 @f8(i32 %a, ptr %dest) {
; CHECK-LABEL: f8:
; CHECK: msfi %r2, -2147483647
; CHECK: br %r14
  %mul = mul i32 %a, 2147483649
  ret i32 %mul
}

; Check multiplication by -1, which is a negation.
define i32 @f9(i32 %a, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK: lcr %r2, %r2
; CHECK: br %r14
  %mul = mul i32 %a, -1
  ret i32 %mul
}

; Check multiplication by -2, which should use shifts.
define i32 @f10(i32 %a, ptr %dest) {
; CHECK-LABEL: f10:
; CHECK: sll %r2, 1
; CHECK: lcr %r2, %r2
; CHECK: br %r14
  %mul = mul i32 %a, -2
  ret i32 %mul
}

; Check multiplication by -3.
define i32 @f11(i32 %a, ptr %dest) {
; CHECK-LABEL: f11:
; CHECK: mhi %r2, -3
; CHECK: br %r14
  %mul = mul i32 %a, -3
  ret i32 %mul
}

; Check the lowest useful MHI value.
define i32 @f12(i32 %a, ptr %dest) {
; CHECK-LABEL: f12:
; CHECK: mhi %r2, -32767
; CHECK: br %r14
  %mul = mul i32 %a, -32767
  ret i32 %mul
}

; Check the next value down, which should use shifts.
define i32 @f13(i32 %a, ptr %dest) {
; CHECK-LABEL: f13:
; CHECK: sll %r2, 15
; CHECK: lcr %r2, %r2
; CHECK: br %r14
  %mul = mul i32 %a, -32768
  ret i32 %mul
}

; Check the next value down again, which can use MSFI.
define i32 @f14(i32 %a, ptr %dest) {
; CHECK-LABEL: f14:
; CHECK: msfi %r2, -32769
; CHECK: br %r14
  %mul = mul i32 %a, -32769
  ret i32 %mul
}

; Check the lowest useful MSFI value.
define i32 @f15(i32 %a, ptr %dest) {
; CHECK-LABEL: f15:
; CHECK: msfi %r2, -2147483647
; CHECK: br %r14
  %mul = mul i32 %a, -2147483647
  ret i32 %mul
}

; Check the next value down, which should use shifts.
define i32 @f16(i32 %a, ptr %dest) {
; CHECK-LABEL: f16:
; CHECK: sll %r2, 31
; CHECK-NOT: lcr
; CHECK: br %r14
  %mul = mul i32 %a, -2147483648
  ret i32 %mul
}

; Check the next value down again, which is treated as a positive value.
define i32 @f17(i32 %a, ptr %dest) {
; CHECK-LABEL: f17:
; CHECK: msfi %r2, 2147483647
; CHECK: br %r14
  %mul = mul i32 %a, -2147483649
  ret i32 %mul
}
