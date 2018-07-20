; Test 64-bit ANDs in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i64 @foo()

; Check NGR.
define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: ngr %r2, %r3
; CHECK: br %r14
  %and = and i64 %a, %b
  ret i64 %and
}

; Check NG with no displacement.
define i64 @f2(i64 %a, i64 *%src) {
; CHECK-LABEL: f2:
; CHECK: ng %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %and = and i64 %a, %b
  ret i64 %and
}

; Check the high end of the aligned NG range.
define i64 @f3(i64 %a, i64 *%src) {
; CHECK-LABEL: f3:
; CHECK: ng %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 *%src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: ng %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check the high end of the negative aligned NG range.
define i64 @f5(i64 %a, i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: ng %r2, -8(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check the low end of the NG range.
define i64 @f6(i64 %a, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: ng %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r3, -524296
; CHECK: ng %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check that NG allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: ng %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %and = and i64 %a, %b
  ret i64 %and
}

; Check that ANDs of spilled values can use NG rather than NGR.
define i64 @f9(i64 *%ptr0) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, foo@PLT
; CHECK: ng %r2, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, i64 *%ptr0, i64 2
  %ptr2 = getelementptr i64, i64 *%ptr0, i64 4
  %ptr3 = getelementptr i64, i64 *%ptr0, i64 6
  %ptr4 = getelementptr i64, i64 *%ptr0, i64 8
  %ptr5 = getelementptr i64, i64 *%ptr0, i64 10
  %ptr6 = getelementptr i64, i64 *%ptr0, i64 12
  %ptr7 = getelementptr i64, i64 *%ptr0, i64 14
  %ptr8 = getelementptr i64, i64 *%ptr0, i64 16
  %ptr9 = getelementptr i64, i64 *%ptr0, i64 18

  %val0 = load i64, i64 *%ptr0
  %val1 = load i64, i64 *%ptr1
  %val2 = load i64, i64 *%ptr2
  %val3 = load i64, i64 *%ptr3
  %val4 = load i64, i64 *%ptr4
  %val5 = load i64, i64 *%ptr5
  %val6 = load i64, i64 *%ptr6
  %val7 = load i64, i64 *%ptr7
  %val8 = load i64, i64 *%ptr8
  %val9 = load i64, i64 *%ptr9

  %ret = call i64 @foo()

  %and0 = and i64 %ret, %val0
  %and1 = and i64 %and0, %val1
  %and2 = and i64 %and1, %val2
  %and3 = and i64 %and2, %val3
  %and4 = and i64 %and3, %val4
  %and5 = and i64 %and4, %val5
  %and6 = and i64 %and5, %val6
  %and7 = and i64 %and6, %val7
  %and8 = and i64 %and7, %val8
  %and9 = and i64 %and8, %val9

  ret i64 %and9
}
