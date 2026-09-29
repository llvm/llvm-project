; Test 128-bit floating-point stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check stores with no offset.
define void @f1(i64 %src, double %val) {
; CHECK-LABEL: f1:
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %ptr = inttoptr i64 %src to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the highest aligned offset that allows STD for both halves.
define void @f2(i64 %src, double %val) {
; CHECK-LABEL: f2:
; CHECK: std %f0, 4080(%r2)
; CHECK: std %f2, 4088(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4080
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the next doubleword up, which requires a mixture of STD and STDY.
define void @f3(i64 %src, double %val) {
; CHECK-LABEL: f3:
; CHECK: std %f0, 4088(%r2)
; CHECK: stdy %f2, 4096(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4088
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the next doubleword after that, which requires STDY for both halves.
define void @f4(i64 %src, double %val) {
; CHECK-LABEL: f4:
; CHECK: stdy %f0, 4096(%r2)
; CHECK: stdy %f2, 4104(%r2)
; CHECK: br %r14
  %add = add i64 %src, 4096
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the highest aligned offset that allows STDY for both halves.
define void @f5(i64 %src, double %val) {
; CHECK-LABEL: f5:
; CHECK: stdy %f0, 524272(%r2)
; CHECK: stdy %f2, 524280(%r2)
; CHECK: br %r14
  %add = add i64 %src, 524272
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the next doubleword up, which requires separate address logic.
; Other sequences besides this one would be OK.
define void @f6(i64 %src, double %val) {
; CHECK-LABEL: f6:
; CHECK: lay %r1, 524280(%r2)
; CHECK: std %f0, 0(%r1)
; CHECK: std %f2, 8(%r1)
; CHECK: br %r14
  %add = add i64 %src, 524280
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the highest aligned negative offset, which needs a combination of
; STDY and STD.
define void @f7(i64 %src, double %val) {
; CHECK-LABEL: f7:
; CHECK: stdy %f0, -8(%r2)
; CHECK: std %f2, 0(%r2)
; CHECK: br %r14
  %add = add i64 %src, -8
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the next doubleword down, which requires STDY for both halves.
define void @f8(i64 %src, double %val) {
; CHECK-LABEL: f8:
; CHECK: stdy %f0, -16(%r2)
; CHECK: stdy %f2, -8(%r2)
; CHECK: br %r14
  %add = add i64 %src, -16
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the lowest offset that allows STDY for both halves.
define void @f9(i64 %src, double %val) {
; CHECK-LABEL: f9:
; CHECK: stdy %f0, -524288(%r2)
; CHECK: stdy %f2, -524280(%r2)
; CHECK: br %r14
  %add = add i64 %src, -524288
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check the next doubleword down, which requires separate address logic.
; Other sequences besides this one would be OK.
define void @f10(i64 %src, double %val) {
; CHECK-LABEL: f10:
; CHECK: agfi %r2, -524296
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %add = add i64 %src, -524296
  %ptr = inttoptr i64 %add to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}

; Check that indices are allowed.
define void @f11(i64 %src, i64 %index, double %val) {
; CHECK-LABEL: f11:
; CHECK: std %f0, 4088({{%r2,%r3|%r3,%r2}})
; CHECK: stdy %f2, 4096({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4088
  %ptr = inttoptr i64 %add2 to ptr
  %ext = fpext double %val to fp128
  store fp128 %ext, ptr %ptr
  ret void
}
