; Test 64-bit floating-point stores.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test the low end of the STD range.
define void @f1(ptr %src, double %val) {
; CHECK-LABEL: f1:
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  store double %val, ptr %src
  ret void
}

; Test the high end of the STD range.
define void @f2(ptr %src, double %val) {
; CHECK-LABEL: f2:
; CHECK: std %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 511
  store double %val, ptr %ptr
  ret void
}

; Check the next doubleword up, which should use STDY instead of STD.
define void @f3(ptr %src, double %val) {
; CHECK-LABEL: f3:
; CHECK: stdy %f0, 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 512
  store double %val, ptr %ptr
  ret void
}

; Check the high end of the aligned STDY range.
define void @f4(ptr %src, double %val) {
; CHECK-LABEL: f4:
; CHECK: stdy %f0, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 65535
  store double %val, ptr %ptr
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f5(ptr %src, double %val) {
; CHECK-LABEL: f5:
; CHECK: agfi %r2, 524288
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 65536
  store double %val, ptr %ptr
  ret void
}

; Check the high end of the negative aligned STDY range.
define void @f6(ptr %src, double %val) {
; CHECK-LABEL: f6:
; CHECK: stdy %f0, -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -1
  store double %val, ptr %ptr
  ret void
}

; Check the low end of the STDY range.
define void @f7(ptr %src, double %val) {
; CHECK-LABEL: f7:
; CHECK: stdy %f0, -524288(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -65536
  store double %val, ptr %ptr
  ret void
}

; Check the next doubleword down, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f8(ptr %src, double %val) {
; CHECK-LABEL: f8:
; CHECK: agfi %r2, -524296
; CHECK: std %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %src, i64 -65537
  store double %val, ptr %ptr
  ret void
}

; Check that STD allows an index.
define void @f9(i64 %src, i64 %index, double %val) {
; CHECK-LABEL: f9:
; CHECK: std %f0, 4095({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4095
  %ptr = inttoptr i64 %add2 to ptr
  store double %val, ptr %ptr
  ret void
}

; Check that STDY allows an index.
define void @f10(i64 %src, i64 %index, double %val) {
; CHECK-LABEL: f10:
; CHECK: stdy %f0, 4096({{%r3,%r2|%r2,%r3}})
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 4096
  %ptr = inttoptr i64 %add2 to ptr
  store double %val, ptr %ptr
  ret void
}
