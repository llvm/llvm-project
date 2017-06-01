; Test 64-bit square root.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @llvm.sqrt.f64(double %f)
declare double @sqrt(double)

; Check register square root.
define double @f1(double %val) {
; CHECK-LABEL: f1:
; CHECK: sqdbr %f0, %f0
; CHECK: br %r14
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the low end of the SQDB range.
define double @f2(double *%ptr) {
; CHECK-LABEL: f2:
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %val = load double , double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the high end of the aligned SQDB range.
define double @f3(double *%base) {
; CHECK-LABEL: f3:
; CHECK: sqdb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %val = load double , double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %val = load double , double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %val = load double , double *%ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check that SQDB allows indices.
define double @f6(double *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: sqdb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %val = load double , double *%ptr2
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Test a case where we spill the source of at least one SQDBR.  We want
; to use SQDB if possible.
define void @f7(double *%ptr) {
; CHECK-LABEL: f7:
; CHECK-SCALAR: sqdb {{%f[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %val0 = load volatile double , double *%ptr
  %val1 = load volatile double , double *%ptr
  %val2 = load volatile double , double *%ptr
  %val3 = load volatile double , double *%ptr
  %val4 = load volatile double , double *%ptr
  %val5 = load volatile double , double *%ptr
  %val6 = load volatile double , double *%ptr
  %val7 = load volatile double , double *%ptr
  %val8 = load volatile double , double *%ptr
  %val9 = load volatile double , double *%ptr
  %val10 = load volatile double , double *%ptr
  %val11 = load volatile double , double *%ptr
  %val12 = load volatile double , double *%ptr
  %val13 = load volatile double , double *%ptr
  %val14 = load volatile double , double *%ptr
  %val15 = load volatile double , double *%ptr
  %val16 = load volatile double , double *%ptr

  %sqrt0 = call double @llvm.sqrt.f64(double %val0)
  %sqrt1 = call double @llvm.sqrt.f64(double %val1)
  %sqrt2 = call double @llvm.sqrt.f64(double %val2)
  %sqrt3 = call double @llvm.sqrt.f64(double %val3)
  %sqrt4 = call double @llvm.sqrt.f64(double %val4)
  %sqrt5 = call double @llvm.sqrt.f64(double %val5)
  %sqrt6 = call double @llvm.sqrt.f64(double %val6)
  %sqrt7 = call double @llvm.sqrt.f64(double %val7)
  %sqrt8 = call double @llvm.sqrt.f64(double %val8)
  %sqrt9 = call double @llvm.sqrt.f64(double %val9)
  %sqrt10 = call double @llvm.sqrt.f64(double %val10)
  %sqrt11 = call double @llvm.sqrt.f64(double %val11)
  %sqrt12 = call double @llvm.sqrt.f64(double %val12)
  %sqrt13 = call double @llvm.sqrt.f64(double %val13)
  %sqrt14 = call double @llvm.sqrt.f64(double %val14)
  %sqrt15 = call double @llvm.sqrt.f64(double %val15)
  %sqrt16 = call double @llvm.sqrt.f64(double %val16)

  store volatile double %val0, double *%ptr
  store volatile double %val1, double *%ptr
  store volatile double %val2, double *%ptr
  store volatile double %val3, double *%ptr
  store volatile double %val4, double *%ptr
  store volatile double %val5, double *%ptr
  store volatile double %val6, double *%ptr
  store volatile double %val7, double *%ptr
  store volatile double %val8, double *%ptr
  store volatile double %val9, double *%ptr
  store volatile double %val10, double *%ptr
  store volatile double %val11, double *%ptr
  store volatile double %val12, double *%ptr
  store volatile double %val13, double *%ptr
  store volatile double %val14, double *%ptr
  store volatile double %val15, double *%ptr
  store volatile double %val16, double *%ptr

  store volatile double %sqrt0, double *%ptr
  store volatile double %sqrt1, double *%ptr
  store volatile double %sqrt2, double *%ptr
  store volatile double %sqrt3, double *%ptr
  store volatile double %sqrt4, double *%ptr
  store volatile double %sqrt5, double *%ptr
  store volatile double %sqrt6, double *%ptr
  store volatile double %sqrt7, double *%ptr
  store volatile double %sqrt8, double *%ptr
  store volatile double %sqrt9, double *%ptr
  store volatile double %sqrt10, double *%ptr
  store volatile double %sqrt11, double *%ptr
  store volatile double %sqrt12, double *%ptr
  store volatile double %sqrt13, double *%ptr
  store volatile double %sqrt14, double *%ptr
  store volatile double %sqrt15, double *%ptr
  store volatile double %sqrt16, double *%ptr

  ret void
}

; Check that a call to the normal sqrt function is lowered.
define double @f8(double %dummy, double %val) {
; CHECK-LABEL: f8:
; CHECK: sqdbr %f0, %f2
; CHECK: cdbr %f0, %f0
; CHECK: bnor %r14
; CHECK: ldr %f0, %f2
; CHECK: jg sqrt@PLT
  %res = tail call double @sqrt(double %val)
  ret double %res
}
