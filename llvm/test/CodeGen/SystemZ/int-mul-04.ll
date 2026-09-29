; Test 64-bit addition in which the second operand is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check MSGR.
define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: msgr %r2, %r3
; CHECK: br %r14
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check MSG with no displacement.
define i64 @f2(i64 %a, ptr %src) {
; CHECK-LABEL: f2:
; CHECK: msg %r2, 0(%r3)
; CHECK: br %r14
  %b = load i64, ptr %src
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check the high end of the aligned MSG range.
define i64 @f3(i64 %a, ptr %src) {
; CHECK-LABEL: f3:
; CHECK: msg %r2, 524280(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65535
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, ptr %src) {
; CHECK-LABEL: f4:
; CHECK: agfi %r3, 524288
; CHECK: msg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 65536
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check the high end of the negative aligned MSG range.
define i64 @f5(i64 %a, ptr %src) {
; CHECK-LABEL: f5:
; CHECK: msg %r2, -8(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -1
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check the low end of the MSG range.
define i64 @f6(i64 %a, ptr %src) {
; CHECK-LABEL: f6:
; CHECK: msg %r2, -524288(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65536
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 %a, ptr %src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r3, -524296
; CHECK: msg %r2, 0(%r3)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %src, i64 -65537
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check that MSG allows an index.
define i64 @f8(i64 %a, i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: msg %r2, 524280({{%r4,%r3|%r3,%r4}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524280
  %ptr = inttoptr i64 %add2 to ptr
  %b = load i64, ptr %ptr
  %mul = mul i64 %a, %b
  ret i64 %mul
}

; Check that multiplications of spilled values can use MSG rather than MSGR.
define i64 @f9(ptr %ptr0) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, foo@PLT
; CHECK: msg %r2, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i64, ptr %ptr0, i64 2
  %ptr2 = getelementptr i64, ptr %ptr0, i64 4
  %ptr3 = getelementptr i64, ptr %ptr0, i64 6
  %ptr4 = getelementptr i64, ptr %ptr0, i64 8
  %ptr5 = getelementptr i64, ptr %ptr0, i64 10
  %ptr6 = getelementptr i64, ptr %ptr0, i64 12
  %ptr7 = getelementptr i64, ptr %ptr0, i64 14
  %ptr8 = getelementptr i64, ptr %ptr0, i64 16
  %ptr9 = getelementptr i64, ptr %ptr0, i64 18

  %val0 = load i64, ptr %ptr0
  %val1 = load i64, ptr %ptr1
  %val2 = load i64, ptr %ptr2
  %val3 = load i64, ptr %ptr3
  %val4 = load i64, ptr %ptr4
  %val5 = load i64, ptr %ptr5
  %val6 = load i64, ptr %ptr6
  %val7 = load i64, ptr %ptr7
  %val8 = load i64, ptr %ptr8
  %val9 = load i64, ptr %ptr9

  %ret = call i64 @foo()

  %mul0 = mul i64 %ret, %val0
  %mul1 = mul i64 %mul0, %val1
  %mul2 = mul i64 %mul1, %val2
  %mul3 = mul i64 %mul2, %val3
  %mul4 = mul i64 %mul3, %val4
  %mul5 = mul i64 %mul4, %val5
  %mul6 = mul i64 %mul5, %val6
  %mul7 = mul i64 %mul6, %val7
  %mul8 = mul i64 %mul7, %val8
  %mul9 = mul i64 %mul8, %val9

  ret i64 %mul9
}
