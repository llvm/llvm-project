; Test conversion of floating-point values to signed i32s.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test f32->i32.
define i32 @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: cfebr %r2, 5, %f0
; CHECK: br %r14
  %conv = fptosi float %f to i32
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: cfdbr %r2, 5, %f0
; CHECK: br %r14
  %conv = fptosi double %f to i32
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(fp128 *%src) {
; CHECK-LABEL: f3:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: cfxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = fptosi fp128 %f to i32
  ret i32 %conv
}
