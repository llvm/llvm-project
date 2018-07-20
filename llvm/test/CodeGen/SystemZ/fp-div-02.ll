; Test 64-bit floating-point division.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @foo()

; Check register division.
define double @f1(double %f1, double %f2) {
; CHECK-LABEL: f1:
; CHECK: ddbr %f0, %f2
; CHECK: br %r14
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the low end of the DDB range.
define double @f2(double %f1, double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double, double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the high end of the aligned DDB range.
define double @f3(double %f1, double *%base) {
; CHECK-LABEL: f3:
; CHECK: ddb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %f2 = load double, double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, double *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %f2 = load double, double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, double *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %f2 = load double, double *%ptr
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check that DDB allows indices.
define double @f6(double %f1, double *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: ddb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %f2 = load double, double *%ptr2
  %res = fdiv double %f1, %f2
  ret double %res
}

; Check that divisions of spilled values can use DDB rather than DDBR.
define double @f7(double *%ptr0) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: ddb %f0, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%ptr0, i64 2
  %ptr2 = getelementptr double, double *%ptr0, i64 4
  %ptr3 = getelementptr double, double *%ptr0, i64 6
  %ptr4 = getelementptr double, double *%ptr0, i64 8
  %ptr5 = getelementptr double, double *%ptr0, i64 10
  %ptr6 = getelementptr double, double *%ptr0, i64 12
  %ptr7 = getelementptr double, double *%ptr0, i64 14
  %ptr8 = getelementptr double, double *%ptr0, i64 16
  %ptr9 = getelementptr double, double *%ptr0, i64 18
  %ptr10 = getelementptr double, double *%ptr0, i64 20

  %val0 = load double, double *%ptr0
  %val1 = load double, double *%ptr1
  %val2 = load double, double *%ptr2
  %val3 = load double, double *%ptr3
  %val4 = load double, double *%ptr4
  %val5 = load double, double *%ptr5
  %val6 = load double, double *%ptr6
  %val7 = load double, double *%ptr7
  %val8 = load double, double *%ptr8
  %val9 = load double, double *%ptr9
  %val10 = load double, double *%ptr10

  %ret = call double @foo()

  %div0 = fdiv double %ret, %val0
  %div1 = fdiv double %div0, %val1
  %div2 = fdiv double %div1, %val2
  %div3 = fdiv double %div2, %val3
  %div4 = fdiv double %div3, %val4
  %div5 = fdiv double %div4, %val5
  %div6 = fdiv double %div5, %val6
  %div7 = fdiv double %div6, %val7
  %div8 = fdiv double %div7, %val8
  %div9 = fdiv double %div8, %val9
  %div10 = fdiv double %div9, %val10

  ret double %div10
}
