; Test 32-bit subtraction in which the second operand is a sign-extended
; i16 memory value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the SH range.
define i32 @f1(i32 %lhs, ptr %src) {
; CHECK-LABEL: f1:
; CHECK: sh %r2, 0(%r3)
; CHECK: br %r14
  %half = load i16, ptr %src
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the high end of the aligned SH range.
define i32 @f2(i32 %lhs, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: sh %r2, 4094(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2047
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the next halfword up, which should use SHY instead of SH.
define i32 @f3(i32 %lhs, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: shy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 2048
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the high end of the aligned SHY range.
define i32 @f4(i32 %lhs, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: shy %r2, 524286(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262143
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the next halfword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f5(i32 %lhs, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: agfi %r3, 524288
; CHECK: sh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 262144
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the high end of the negative aligned SHY range.
define i32 @f6(i32 %lhs, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: shy %r2, -2(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -1
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the low end of the SHY range.
define i32 @f7(i32 %lhs, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: shy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262144
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check the next halfword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f8(i32 %lhs, ptr %src) {
; CHECK-LABEL: f8:
; CHECK: agfi %r3, -524290
; CHECK: sh %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %src, i64 -262145
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check that SH allows an index.
define i32 @f9(i32 %lhs, i64 %src, i64 %index) {
; CHECK-LABEL: f9:
; CHECK: sh %r2, 4094({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %sub1 = add i64 %src, %index
  %sub2 = add i64 %sub1, 4094
  %ptr = inttoptr i64 %sub2 to ptr
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}

; Check that SHY allows an index.
define i32 @f10(i32 %lhs, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: shy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %sub1 = add i64 %src, %index
  %sub2 = add i64 %sub1, 4096
  %ptr = inttoptr i64 %sub2 to ptr
  %half = load i16, ptr %ptr
  %rhs = sext i16 %half to i32
  %res = sub i32 %lhs, %rhs
  ret i32 %res
}
