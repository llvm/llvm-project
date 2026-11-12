; Test moves of integers to 2-byte memory locations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check the low end of the unsigned range.
define void @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: mvhhi 0(%r2), 0
; CHECK: br %r14
  store i16 0, ptr %ptr
  ret void
}

; Check the high end of the signed range.
define void @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: mvhhi 0(%r2), 32767
; CHECK: br %r14
  store i16 32767, ptr %ptr
  ret void
}

; Check the next value up.
define void @f3(ptr %ptr) {
; CHECK-LABEL: f3:
; CHECK: mvhhi 0(%r2), -32768
; CHECK: br %r14
  store i16 -32768, ptr %ptr
  ret void
}

; Check the high end of the unsigned range.
define void @f4(ptr %ptr) {
; CHECK-LABEL: f4:
; CHECK: mvhhi 0(%r2), -1
; CHECK: br %r14
  store i16 65535, ptr %ptr
  ret void
}

; Check -1.
define void @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: mvhhi 0(%r2), -1
; CHECK: br %r14
  store i16 -1, ptr %ptr
  ret void
}

; Check the low end of the signed range.
define void @f6(ptr %ptr) {
; CHECK-LABEL: f6:
; CHECK: mvhhi 0(%r2), -32768
; CHECK: br %r14
  store i16 -32768, ptr %ptr
  ret void
}

; Check the next value down.
define void @f7(ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK: mvhhi 0(%r2), 32767
; CHECK: br %r14
  store i16 -32769, ptr %ptr
  ret void
}

; Check the high end of the MVHHI range.
define void @f8(ptr %a) {
; CHECK-LABEL: f8:
; CHECK: mvhhi 4094(%r2), 42
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %a, i64 2047
  store i16 42, ptr %ptr
  ret void
}

; Check the next halfword up, which is out of range.  We prefer STHY
; in that case.
define void @f9(ptr %a) {
; CHECK-LABEL: f9:
; CHECK: lhi [[TMP:%r[0-5]]], 42
; CHECK: sthy [[TMP]], 4096(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %a, i64 2048
  store i16 42, ptr %ptr
  ret void
}

; Check negative displacements, for which we again prefer STHY.
define void @f10(ptr %a) {
; CHECK-LABEL: f10:
; CHECK: lhi [[TMP:%r[0-5]]], 42
; CHECK: sthy [[TMP]], -2(%r2)
; CHECK: br %r14
  %ptr = getelementptr i16, ptr %a, i64 -1
  store i16 42, ptr %ptr
  ret void
}

; Check that MVHHI does not allow an index.
define void @f11(i64 %src, i64 %index) {
; CHECK-LABEL: f11:
; CHECK: lhi [[TMP:%r[0-5]]], 42
; CHECK: sth [[TMP]], 0({{%r2,%r3|%r3,%r2}})
; CHECK: br %r14
  %add = add i64 %src, %index
  %ptr = inttoptr i64 %add to ptr
  store i16 42, ptr %ptr
  ret void
}
