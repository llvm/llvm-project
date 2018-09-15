; Test conversion of floating-point values to signed i64s.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test f32->i64.
define i64 @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: cgebr %r2, 5, %f0
; CHECK: br %r14
  %conv = fptosi float %f to i64
  ret i64 %conv
}

; Test f64->i64.
define i64 @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: cgdbr %r2, 5, %f0
; CHECK: br %r14
  %conv = fptosi double %f to i64
  ret i64 %conv
}

; Test f128->i64.
define i64 @f3(fp128 *%src) {
; CHECK-LABEL: f3:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: cgxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = fptosi fp128 %f to i64
  ret i64 %conv
}
