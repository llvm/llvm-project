; Test extensions of f32 to f64.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Check register extension.
define double @f1(float %val) {
; CHECK-LABEL: f1:
; CHECK: ldebr %f0, %f0
; CHECK: br %r14
  %res = fpext float %val to double
  ret double %res
}

; Check the low end of the LDEB range.
define double @f2(float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %val = load float , float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check the high end of the aligned LDEB range.
define double @f3(float *%base) {
; CHECK-LABEL: f3:
; CHECK: ldeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %val = load float , float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %val = load float , float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: ldeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %val = load float , float *%ptr
  %res = fpext float %val to double
  ret double %res
}

; Check that LDEB allows indices.
define double @f6(float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: ldeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %val = load float , float *%ptr2
  %res = fpext float %val to double
  ret double %res
}

