; Test moves of integers to 8-byte memory locations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check moves of zero.
define void @f1(ptr %a) {
; CHECK-LABEL: f1:
; CHECK: mvghi 0(%r2), 0
; CHECK: br %r14
  store i64 0, ptr %a
  ret void
}

; Check the high end of the signed 16-bit range.
define void @f2(ptr %a) {
; CHECK-LABEL: f2:
; CHECK: mvghi 0(%r2), 32767
; CHECK: br %r14
  store i64 32767, ptr %a
  ret void
}

; Check the next value up, which can't use MVGHI.
define void @f3(ptr %a) {
; CHECK-LABEL: f3:
; CHECK-NOT: mvghi
; CHECK: br %r14
  store i64 32768, ptr %a
  ret void
}

; Check moves of -1.
define void @f4(ptr %a) {
; CHECK-LABEL: f4:
; CHECK: mvghi 0(%r2), -1
; CHECK: br %r14
  store i64 -1, ptr %a
  ret void
}

; Check the low end of the MVGHI range.
define void @f5(ptr %a) {
; CHECK-LABEL: f5:
; CHECK: mvghi 0(%r2), -32768
; CHECK: br %r14
  store i64 -32768, ptr %a
  ret void
}

; Check the next value down, which can't use MVGHI.
define void @f6(ptr %a) {
; CHECK-LABEL: f6:
; CHECK-NOT: mvghi
; CHECK: br %r14
  store i64 -32769, ptr %a
  ret void
}

; Check the high end of the MVGHI range.
define void @f7(ptr %a) {
; CHECK-LABEL: f7:
; CHECK: mvghi 4088(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %a, i64 511
  store i64 42, ptr %ptr
  ret void
}

; Check the next doubleword up, which is out of range.  We prefer STG
; in that case.
define void @f8(ptr %a) {
; CHECK-LABEL: f8:
; CHECK: lghi [[TMP:%r[0-5]]], 42
; CHECK: stg [[TMP]], 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %a, i64 512
  store i64 42, ptr %ptr
  ret void
}

; Check negative displacements, for which we again prefer STG.
define void @f9(ptr %a) {
; CHECK-LABEL: f9:
; CHECK: lghi [[TMP:%r[0-5]]], 42
; CHECK: stg [[TMP]], -8(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, ptr %a, i64 -1
  store i64 42, ptr %ptr
  ret void
}

; Check that MVGHI does not allow an index.
define void @f10(i64 %src, i64 %index) {
; CHECK-LABEL: f10:
; CHECK: lghi [[TMP:%r[0-5]]], 42
; CHECK: stg [[TMP]], 0({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add = add i64 %src, %index
  %ptr = inttoptr i64 %add to ptr
  store i64 42, ptr %ptr
  ret void
}
