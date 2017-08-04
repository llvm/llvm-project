; Test 32-bit floating-point addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @foo()

; Check register addition.
define float @f1(float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: aebr %f0, %f2
; CHECK: br %r14
  %res = fadd float %f1, %f2
  ret float %res
}

; Check the low end of the AEB range.
define float @f2(float %f1, float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: aeb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float , float *%ptr
  %res = fadd float %f1, %f2
  ret float %res
}

; Check the high end of the aligned AEB range.
define float @f3(float %f1, float *%base) {
; CHECK-LABEL: f3:
; CHECK: aeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %f2 = load float , float *%ptr
  %res = fadd float %f1, %f2
  ret float %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define float @f4(float %f1, float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: aeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %f2 = load float , float *%ptr
  %res = fadd float %f1, %f2
  ret float %res
}

; Check negative displacements, which also need separate address logic.
define float @f5(float %f1, float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: aeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %f2 = load float , float *%ptr
  %res = fadd float %f1, %f2
  ret float %res
}

; Check that AEB allows indices.
define float @f6(float %f1, float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: aeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %f2 = load float , float *%ptr2
  %res = fadd float %f1, %f2
  ret float %res
}

; Check that additions of spilled values can use AEB rather than AEBR.
define float @f7(float *%ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: aeb %f0, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%ptr0, i64 2
  %ptr2 = getelementptr float, float *%ptr0, i64 4
  %ptr3 = getelementptr float, float *%ptr0, i64 6
  %ptr4 = getelementptr float, float *%ptr0, i64 8
  %ptr5 = getelementptr float, float *%ptr0, i64 10
  %ptr6 = getelementptr float, float *%ptr0, i64 12
  %ptr7 = getelementptr float, float *%ptr0, i64 14
  %ptr8 = getelementptr float, float *%ptr0, i64 16
  %ptr9 = getelementptr float, float *%ptr0, i64 18
  %ptr10 = getelementptr float, float *%ptr0, i64 20

  %val0 = load float , float *%ptr0
  %val1 = load float , float *%ptr1
  %val2 = load float , float *%ptr2
  %val3 = load float , float *%ptr3
  %val4 = load float , float *%ptr4
  %val5 = load float , float *%ptr5
  %val6 = load float , float *%ptr6
  %val7 = load float , float *%ptr7
  %val8 = load float , float *%ptr8
  %val9 = load float , float *%ptr9
  %val10 = load float , float *%ptr10

  %ret = call float @foo()

  %add0 = fadd float %ret, %val0
  %add1 = fadd float %add0, %val1
  %add2 = fadd float %add1, %val2
  %add3 = fadd float %add2, %val3
  %add4 = fadd float %add3, %val4
  %add5 = fadd float %add4, %val5
  %add6 = fadd float %add5, %val6
  %add7 = fadd float %add6, %val7
  %add8 = fadd float %add7, %val8
  %add9 = fadd float %add8, %val9
  %add10 = fadd float %add9, %val10

  ret float %add10
}
