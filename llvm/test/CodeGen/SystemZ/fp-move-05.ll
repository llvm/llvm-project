; Test 128-bit floating-point loads.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -verify-machineinstrs | FileCheck %s

; Check loads with no offset.
define double @f1(i64 %src) {
; CHECK-LABEL: f1:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: br %r14
  %ptr = inttoptr i64 %src to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the highest aligned offset that allows LD for both halves.
define double @f2(i64 %src) {
; CHECK-LABEL: f2:
; CHECK: ld %f0, 4080(%r2)
; CHECK: ld %f2, 4088(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4080
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the next doubleword up, which requires a mixture of LD and LDY.
define double @f3(i64 %src) {
; CHECK-LABEL: f3:
; CHECK: ld %f0, 4088(%r2)
; CHECK: ldy %f2, 4096(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4088
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the next doubleword after that, which requires LDY for both halves.
define double @f4(i64 %src) {
; CHECK-LABEL: f4:
; CHECK: ldy %f0, 4096(%r2)
; CHECK: ldy %f2, 4104(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4096
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the highest aligned offset that allows LDY for both halves.
define double @f5(i64 %src) {
; CHECK-LABEL: f5:
; CHECK: ldy %f0, 524272(%r2)
; CHECK: ldy %f2, 524280(%r2)
; CHECK: br %r14
  %add = add i64 %src, 524272
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the next doubleword up, which requires separate address logic.
; Other sequences besides this one would be OK.
define double @f6(i64 %src) {
; CHECK-LABEL: f6:
; CHECK: lay %r1, 524280(%r2)
; CHECK: ld %f0, 0(%r1)
; CHECK: ld %f2, 8(%r1)
; CHECK: br %r14
  %add = add i64 %src, 524280
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the highest aligned negative offset, which needs a combination of
; LDY and LD.
define double @f7(i64 %src) {
; CHECK-LABEL: f7:
; CHECK: ldy %f0, -8(%r2)
; CHECK: ld %f2, 0(%r2)
; CHECK: br %r14
  %add = add i64 %src, -8
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the next doubleword down, which requires LDY for both halves.
define double @f8(i64 %src) {
; CHECK-LABEL: f8:
; CHECK: ldy %f0, -16(%r2)
; CHECK: ldy %f2, -8(%r2)
; CHECK: br %r14
  %add = add i64 %src, -16
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the lowest offset that allows LDY for both halves.
define double @f9(i64 %src) {
; CHECK-LABEL: f9:
; CHECK: ldy %f0, -524288(%r2)
; CHECK: ldy %f2, -524280(%r2)
; CHECK: br %r14
  %add = add i64 %src, -524288
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check the next doubleword down, which requires separate address logic.
; Other sequences besides this one would be OK.
define double @f10(i64 %src) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524296
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: br %r14
  %add = add i64 %src, -524296
  %ptr = inttoptr i64 %add to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}

; Check that indices are allowed.
define double @f11(i64 %src, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: ld %f0, 4088({{%r2,%r3|%r3,%r2}})
; CHECK: ldy %f2, 4096({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to fp128 *
  %val = load fp128, fp128 *%ptr
  %trunc = fptrunc fp128 %val to double
  ret double %trunc
}
