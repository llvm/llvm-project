; Test conversions of unsigned integers to floating-point values
; (z196 and above).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check i32->f32.
define float @f1(i32 %i) {
; CHECK-LABEL: f1:
; CHECK: celfbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = uitofp i32 %i to float
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) {
; CHECK-LABEL: f2:
; CHECK: cdlfbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = uitofp i32 %i to double
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) {
; CHECK-LABEL: f3:
; CHECK: cxlfbr %f0, 0, %r2, 0
; CHECK-DAG: std %f0, 0(%r3)
; CHECK-DAG: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = uitofp i32 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}

; Check i64->f32.
define float @f4(i64 %i) {
; CHECK-LABEL: f4:
; CHECK: celgbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = uitofp i64 %i to float
  ret float %conv
}

; Check i64->f64.
define double @f5(i64 %i) {
; CHECK-LABEL: f5:
; CHECK: cdlgbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = uitofp i64 %i to double
  ret double %conv
}

; Check i64->f128.
define void @f6(i64 %i, ptr %dst) {
; CHECK-LABEL: f6:
; CHECK: cxlgbr %f0, 0, %r2, 0
; CHECK-DAG: std %f0, 0(%r3)
; CHECK-DAG: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = uitofp i64 %i to fp128
  store fp128 %conv, ptr %dst
  ret void
}
