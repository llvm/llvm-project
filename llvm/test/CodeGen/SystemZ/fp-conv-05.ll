; Test conversions of signed i32s to floating-point values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Check i32->f32.
define float @f1(i32 %i) {
; CHECK-LABEL: f1:
; CHECK: cefbr %f0, %r2
; CHECK: br %r14
  %conv = sitofp i32 %i to float
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) {
; CHECK-LABEL: f2:
; CHECK: cdfbr %f0, %r2
; CHECK: br %r14
  %conv = sitofp i32 %i to double
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: cxfbr %f0, %r2
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = sitofp i32 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}
