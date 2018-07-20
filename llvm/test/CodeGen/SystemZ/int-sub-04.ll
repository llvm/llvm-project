; Test 64-bit subtraction in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i64 @foo()

; Check SGR.
define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: sgr %r2, %r3
; CHECK: br %r14
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check SG with no displacement.
define i64 @f2(i64 %a, i64 *%src) {
; CHECK-LABEL: f2:
; CHECK: sg %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64, i64 *%src
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check the high end of the aligned SG range.
define i64 @f3(i64 %a, i64 *%src) {
; CHECK-LABEL: f3:
; CHECK: sg %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 *%src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: sg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check the high end of the negative aligned SG range.
define i64 @f5(i64 %a, i64 *%src) {
; CHECK-LABEL: f5:
; CHECK: sg %r2, -8(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -1
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check the low end of the SG range.
define i64 @f6(i64 %a, i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: sg %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65536
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r3, -524296
; CHECK: sg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 -65537
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check that SG allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: sg %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to i64 *
  %b = load i64, i64 *%ptr
  %sub = sub i64 %a, %b
  ret i64 %sub
}

; Check that subtractions of spilled values can use SG rather than SGR.
define i64 @f9(i64 *%ptr0) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, foo@PLT
; CHECK: sg %r2, 160(%r15)
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

  %sub0 = sub i64 %ret, %val0
  %sub1 = sub i64 %sub0, %val1
  %sub2 = sub i64 %sub1, %val2
  %sub3 = sub i64 %sub2, %val3
  %sub4 = sub i64 %sub3, %val4
  %sub5 = sub i64 %sub4, %val5
  %sub6 = sub i64 %sub5, %val6
  %sub7 = sub i64 %sub6, %val7
  %sub8 = sub i64 %sub7, %val8
  %sub9 = sub i64 %sub8, %val9

  ret i64 %sub9
}
