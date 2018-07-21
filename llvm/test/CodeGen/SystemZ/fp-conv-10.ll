; Test conversion of floating-point values to unsigned i32s (z10 only).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; z10 doesn't have native support for unsigned fp-to-i32 conversions;
; they were added in z196 as the Convert to Logical family of instructions.
; Promoting to i64 doesn't generate an inexact condition for values that are
; outside the i32 range but in the i64 range, so use the default expansion.

; Test f32->i32.
define i32 @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: cebr
; CHECK: sebr
; CHECK: cfebr
; CHECK: xilf
; CHECK: br %r14
  %conv = fptoui float %f to i32
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: cdbr
; CHECK: sdbr
; CHECK: cfdbr
; CHECK: xilf
; CHECK: br %r14
  %conv = fptoui double %f to i32
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(fp128 *%src) {
; CHECK-LABEL: f3:
; CHECK: cxbr
; CHECK: sxbr
; CHECK: cfxbr
; CHECK: xilf
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = fptoui fp128 %f to i32
  ret i32 %conv
}
