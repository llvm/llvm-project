; Test that we take advantage of signext and zeroext annotations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Zero extension of something that is already zero-extended.
define void @f1(i32 zeroext %r2, ptr %r3) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK: stg %r2, 0(%r3)
; CHECK: br %r14
  %conv = zext i32 %r2 to i64
  store i64 %conv, ptr %r3
  ret void
}

; Sign extension of something that is already sign-extended.
define void @f2(i32 signext %r2, ptr %r3) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK: stg %r2, 0(%r3)
; CHECK: br %r14
  %conv = sext i32 %r2 to i64
  store i64 %conv, ptr %r3
  ret void
}

; Sign extension of something that is already zero-extended.
define void @f3(i32 zeroext %r2, ptr %r3) {
; CHECK-LABEL: f3:
; CHECK: lgfr [[REGISTER:%r[0-5]+]], %r2
; CHECK: stg [[REGISTER]], 0(%r3)
; CHECK: br %r14
  %conv = sext i32 %r2 to i64
  store i64 %conv, ptr %r3
  ret void
}

; Zero extension of something that is already sign-extended.
define void @f4(i32 signext %r2, ptr %r3) {
; CHECK-LABEL: f4:
; CHECK: llgfr [[REGISTER:%r[0-5]+]], %r2
; CHECK: stg [[REGISTER]], 0(%r3)
; CHECK: br %r14
  %conv = zext i32 %r2 to i64
  store i64 %conv, ptr %r3
  ret void
}
