; Test memory-to-memory XORs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test the simple i8 case.
define void @f1(ptr %ptr1) {
; CHECK-LABEL: f1:
; CHECK: xc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  %old = load i8, ptr %ptr2
  %xor = xor i8 %val, %old
  store i8 %xor, ptr %ptr2
  ret void
}

; Test the simple i16 case.
define void @f2(ptr %ptr1) {
; CHECK-LABEL: f2:
; CHECK: xc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  %old = load i16, ptr %ptr2
  %xor = xor i16 %val, %old
  store i16 %xor, ptr %ptr2
  ret void
}

; Test the simple i32 case.
define void @f3(ptr %ptr1) {
; CHECK-LABEL: f3:
; CHECK: xc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, ptr %ptr1, i64 1
  %val = load i32, ptr %ptr1
  %old = load i32, ptr %ptr2
  %xor = xor i32 %old, %val
  store i32 %xor, ptr %ptr2
  ret void
}

; Test the i64 case.
define void @f4(ptr %ptr1) {
; CHECK-LABEL: f4:
; CHECK: xc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64, ptr %ptr1, i64 1
  %val = load i64, ptr %ptr1
  %old = load i64, ptr %ptr2
  %xor = xor i64 %old, %val
  store i64 %xor, ptr %ptr2
  ret void
}

; Leave other more complicated tests to and-08.ll.
