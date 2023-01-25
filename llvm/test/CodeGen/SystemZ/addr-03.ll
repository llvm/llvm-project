; Test constant addresses, unlikely as they are.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1() {
; CHECK-LABEL: f1:
; CHECK: lb %r0, 0
; CHECK: br %r14
  %ptr = inttoptr i64 0 to ptr
  %val = load volatile i8, ptr %ptr
  ret void
}

define void @f2() {
; CHECK-LABEL: f2:
; CHECK: lb %r0, -524288
; CHECK: br %r14
  %ptr = inttoptr i64 -524288 to ptr
  %val = load volatile i8, ptr %ptr
  ret void
}

define void @f3() {
; CHECK-LABEL: f3:
; CHECK-NOT: lb %r0, -524289
; CHECK: br %r14
  %ptr = inttoptr i64 -524289 to ptr
  %val = load volatile i8, ptr %ptr
  ret void
}

define void @f4() {
; CHECK-LABEL: f4:
; CHECK: lb %r0, 524287
; CHECK: br %r14
  %ptr = inttoptr i64 524287 to ptr
  %val = load volatile i8, ptr %ptr
  ret void
}

define void @f5() {
; CHECK-LABEL: f5:
; CHECK-NOT: lb %r0, 524288
; CHECK: br %r14
  %ptr = inttoptr i64 524288 to ptr
  %val = load volatile i8, ptr %ptr
  ret void
}
