; Test 32-bit multiplication in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @foo()

; Check MSR.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: msr %r2, %r3
; CHECK: br %r14
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the low end of the MS range.
define i32 @f2(i32 %a, i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: ms %r2, 0(%r3)
; CHECK: br %r14
  %b = load i32, i32 *%src
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the high end of the aligned MS range.
define i32 @f3(i32 %a, i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: ms %r2, 4092(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1023
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the next word up, which should use MSY instead of MS.
define i32 @f4(i32 %a, i32 *%src) {
; CHECK-LABEL: f4:
; CHECK: msy %r2, 4096(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 1024
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the high end of the aligned MSY range.
define i32 @f5(i32 %a, i32 *%src) {
; CHECK-LABEL: f5:
; CHECK: msy %r2, 524284(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f6(i32 %a, i32 *%src) {
; CHECK-LABEL: f6:
; CHECK: agfi %r3, 524288
; CHECK: ms %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the high end of the negative aligned MSY range.
define i32 @f7(i32 %a, i32 *%src) {
; CHECK-LABEL: f7:
; CHECK: msy %r2, -4(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -1
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the low end of the MSY range.
define i32 @f8(i32 %a, i32 *%src) {
; CHECK-LABEL: f8:
; CHECK: msy %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131072
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f9(i32 %a, i32 *%src) {
; CHECK-LABEL: f9:
; CHECK: agfi %r3, -524292
; CHECK: ms %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 -131073
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check that MS allows an index.
define i32 @f10(i32 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: ms %r2, 4092({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4092
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check that MSY allows an index.
define i32 @f11(i32 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: msy %r2, 4096({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to i32 *
  %b = load i32, i32 *%ptr
  %mul = mul i32 %a, %b
  ret i32 %mul
}

; Check that multiplications of spilled values can use MS rather than MSR.
define i32 @f12(i32 *%ptr0) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, foo@PLT
; CHECK: ms %r2, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16
  %ptr9 = getelementptr i32, i32 *%ptr0, i64 18

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6
  %val7 = load i32, i32 *%ptr7
  %val8 = load i32, i32 *%ptr8
  %val9 = load i32, i32 *%ptr9

  %ret = call i32 @foo()

  %mul0 = mul i32 %ret, %val0
  %mul1 = mul i32 %mul0, %val1
  %mul2 = mul i32 %mul1, %val2
  %mul3 = mul i32 %mul2, %val3
  %mul4 = mul i32 %mul3, %val4
  %mul5 = mul i32 %mul4, %val5
  %mul6 = mul i32 %mul5, %val6
  %mul7 = mul i32 %mul6, %val7
  %mul8 = mul i32 %mul7, %val8
  %mul9 = mul i32 %mul8, %val9

  ret i32 %mul9
}
