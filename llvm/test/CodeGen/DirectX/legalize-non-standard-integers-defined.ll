; RUN: opt -S -passes='dxil-legalize,instcombine' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Check defined-input results rather than requiring exact poison propagation.

define i32 @signed_add() {
; CHECK-LABEL: define i32 @signed_add(
; CHECK: ret i32 -1
  %sum = add nsw i3 -3, 2
  %result = sext i3 %sum to i32
  ret i32 %result
}

define i32 @unsigned_add() {
; CHECK-LABEL: define i32 @unsigned_add(
; CHECK: ret i32 7
  %sum = add nuw i3 5, 2
  %result = zext i3 %sum to i32
  ret i32 %result
}

define i32 @both_flags() {
; CHECK-LABEL: define i32 @both_flags(
; CHECK: ret i32 7
  %sum = add nuw nsw i3 -3, 2
  %result = zext i3 %sum to i32
  ret i32 %result
}

define i32 @unsigned_consumer_after_signed_mul() {
; CHECK-LABEL: define i32 @unsigned_consumer_after_signed_mul(
; CHECK: ret i32 1
  %product = mul nsw i3 -1, 2
  %quotient = udiv i3 %product, 4
  %result = zext i3 %quotient to i32
  ret i32 %result
}

define i32 @signed_consumer_after_unsigned_mul() {
; CHECK-LABEL: define i32 @signed_consumer_after_unsigned_mul(
; CHECK: ret i32 -1
  %product = mul nuw i3 3, 2
  %quotient = sdiv i3 %product, 2
  %result = sext i3 %quotient to i32
  ret i32 %result
}

define i32 @signed_trunc() {
; CHECK-LABEL: define i32 @signed_trunc(
; CHECK: ret i32 -4
  %narrow = trunc nsw i32 -4 to i3
  %result = sext i3 %narrow to i32
  ret i32 %result
}

define i32 @unsigned_trunc() {
; CHECK-LABEL: define i32 @unsigned_trunc(
; CHECK: ret i32 7
  %narrow = trunc nuw i32 7 to i3
  %result = zext i3 %narrow to i32
  ret i32 %result
}

define i32 @arithmetic_shift() {
; CHECK-LABEL: define i32 @arithmetic_shift(
; CHECK: ret i32 -1
  %shifted = ashr exact i3 -4, 2
  %result = sext i3 %shifted to i32
  ret i32 %result
}

define i32 @logical_shift() {
; CHECK-LABEL: define i32 @logical_shift(
; CHECK: ret i32 1
  %shifted = lshr exact i3 4, 2
  %result = zext i3 %shifted to i32
  ret i32 %result
}

define i32 @left_shift() {
; CHECK-LABEL: define i32 @left_shift(
; CHECK: ret i32 6
  %shifted = shl nuw i3 3, 1
  %result = zext i3 %shifted to i32
  ret i32 %result
}

define i32 @signed_fp_conversion() {
; CHECK-LABEL: define i32 @signed_fp_conversion(
; CHECK: ret i32 -4
  %narrow = fptosi float -4.5 to i3
  %result = sext i3 %narrow to i32
  ret i32 %result
}

define i32 @unsigned_fp_conversion() {
; CHECK-LABEL: define i32 @unsigned_fp_conversion(
; CHECK: ret i32 7
  %narrow = fptoui float 7.5 to i3
  %result = zext i3 %narrow to i32
  ret i32 %result
}
