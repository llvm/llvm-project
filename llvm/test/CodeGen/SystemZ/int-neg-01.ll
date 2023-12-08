; Test integer negation.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test i32->i32 negation.
define i32 @f1(i32 %val) {
; CHECK-LABEL: f1:
; CHECK: lcr %r2, %r2
; CHECK: br %r14
  %neg = sub i32 0, %val
  ret i32 %neg
}

; Test i32->i64 negation.
define i64 @f2(i32 %val) {
; CHECK-LABEL: f2:
; CHECK: lcgfr %r2, %r2
; CHECK: br %r14
  %ext = sext i32 %val to i64
  %neg = sub i64 0, %ext
  ret i64 %neg
}

; Test i32->i64 negation that uses an "in-register" form of sign extension.
define i64 @f3(i64 %val) {
; CHECK-LABEL: f3:
; CHECK: lcgfr %r2, %r2
; CHECK: br %r14
  %trunc = trunc i64 %val to i32
  %ext = sext i32 %trunc to i64
  %neg = sub i64 0, %ext
  ret i64 %neg
}

; Test i64 negation.
define i64 @f4(i64 %val) {
; CHECK-LABEL: f4:
; CHECK: lcgr %r2, %r2
; CHECK: br %r14
  %neg = sub i64 0, %val
  ret i64 %neg
}
