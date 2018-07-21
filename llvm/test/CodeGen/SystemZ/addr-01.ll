; Test selection of addresses with indices in cases where the address
; is used once.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; A simple index address.
define void @f1(i64 %addr, i64 %index) {
; CHECK-LABEL: f1:
; CHECK: lb %r0, 0(%r3,%r2)
; CHECK: br %r14
  %add = add i64 %addr, %index
  %ptr = inttoptr i64 %add to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; An address with an index and a displacement (order 1).
define void @f2(i64 %addr, i64 %index) {
; CHECK-LABEL: f2:
; CHECK: lb %r0, 100(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %addr, %index
  %add2 = add i64 %add1, 100
  %ptr = inttoptr i64 %add2 to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; An address with an index and a displacement (order 2).
define void @f3(i64 %addr, i64 %index) {
; CHECK-LABEL: f3:
; CHECK: lb %r0, 100(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %addr, 100
  %add2 = add i64 %add1, %index
  %ptr = inttoptr i64 %add2 to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; An address with an index and a subtracted displacement (order 1).
define void @f4(i64 %addr, i64 %index) {
; CHECK-LABEL: f4:
; CHECK: lb %r0, -100(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %addr, %index
  %add2 = sub i64 %add1, 100
  %ptr = inttoptr i64 %add2 to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; An address with an index and a subtracted displacement (order 2).
define void @f5(i64 %addr, i64 %index) {
; CHECK-LABEL: f5:
; CHECK: lb %r0, -100(%r3,%r2)
; CHECK: br %r14
  %add1 = sub i64 %addr, 100
  %add2 = add i64 %add1, %index
  %ptr = inttoptr i64 %add2 to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; An address with an index and a displacement added using OR.
define void @f6(i64 %addr, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: nill %r2, 65528
; CHECK: lb %r0, 6(%r3,%r2)
; CHECK: br %r14
  %aligned = and i64 %addr, -8
  %or = or i64 %aligned, 6
  %add = add i64 %or, %index
  %ptr = inttoptr i64 %add to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; Like f6, but without the masking.  This OR doesn't count as a displacement.
define void @f7(i64 %addr, i64 %index) {
; CHECK-LABEL: f7:
; CHECK: oill %r2, 6
; CHECK: lb %r0, 0(%r3,%r2)
; CHECK: br %r14
  %or = or i64 %addr, 6
  %add = add i64 %or, %index
  %ptr = inttoptr i64 %add to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}

; Like f6, but with the OR applied after the index.  We don't know anything
; about the alignment of %add here.
define void @f8(i64 %addr, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: nill %r2, 65528
; CHECK: agr %r2, %r3
; CHECK: oill %r2, 6
; CHECK: lb %r0, 0(%r2)
; CHECK: br %r14
  %aligned = and i64 %addr, -8
  %add = add i64 %aligned, %index
  %or = or i64 %add, 6
  %ptr = inttoptr i64 %or to i8 *
  %a = load volatile i8, i8 *%ptr
  ret void
}
