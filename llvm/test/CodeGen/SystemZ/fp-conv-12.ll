; Test conversion of floating-point values to unsigned i64s (z10 only).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; z10 doesn't have native support for unsigned fp-to-i64 conversions;
; they were added in z196 as the Convert to Logical family of instructions.
; Convert via signed i64s instead.

; Test f32->i64.
define i64 @f1(float %f) {
; CHECK-LABEL: f1:
; CHECK: cebr
; CHECK: sebr
; CHECK: cgebr
; CHECK: xihf
; CHECK: br %r14
  %conv = fptoui float %f to i64
  ret i64 %conv
}

; Test f64->i64.
define i64 @f2(double %f) {
; CHECK-LABEL: f2:
; CHECK: cdbr
; CHECK: sdbr
; CHECK: cgdbr
; CHECK: xihf
; CHECK: br %r14
  %conv = fptoui double %f to i64
  ret i64 %conv
}

; Test f128->i64.
define i64 @f3(fp128 *%src) {
; CHECK-LABEL: f3:
; CHECK: cxbr
; CHECK: sxbr
; CHECK: cgxbr
; CHECK: xihf
; CHECK: br %r14
  %f = load fp128, fp128 *%src
  %conv = fptoui fp128 %f to i64
  ret i64 %conv
}
