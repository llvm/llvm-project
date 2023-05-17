; Test conversions of unsigned i32s to floating-point values (z10 only).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Check i32->f32.  There is no native instruction, so we must promote
; to i64 first.
define float @f1(i32 %i) {
; CHECK-LABEL: f1:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cegbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = uitofp i32 %i to float
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) {
; CHECK-LABEL: f2:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cdgbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = uitofp i32 %i to double
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cxgbr %f0, [[REGISTER]]
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = uitofp i32 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}
