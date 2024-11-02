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
define double @f2(ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %val = load double, ptr %ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the high end of the aligned SQDB range.
define double @f3(ptr %base) {
; CHECK-LABEL: f3:
; CHECK: sqdb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 511
  %val = load double, ptr %ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(ptr %base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 512
  %val = load double, ptr %ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(ptr %base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: sqdb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 -1
  %val = load double, ptr %ptr
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Check that SQDB allows indices.
define double @f6(ptr %base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: sqdb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %base, i64 %index
  %ptr2 = getelementptr double, ptr %ptr1, i64 100
  %val = load double, ptr %ptr2
  %res = call double @llvm.sqrt.f64(double %val)
  ret double %res
}

; Test a case where we spill the source of at least one SQDBR.  We want
; to use SQDB if possible.
define void @f7(ptr %ptr) {
; CHECK-LABEL: f7:
; CHECK-SCALAR: sqdb {{%f[0-9]+}}, 160(%r15)
; CHECK: br %r14
  %val0 = load volatile double, ptr %ptr
  %val1 = load volatile double, ptr %ptr
  %val2 = load volatile double, ptr %ptr
  %val3 = load volatile double, ptr %ptr
  %val4 = load volatile double, ptr %ptr
  %val5 = load volatile double, ptr %ptr
  %val6 = load volatile double, ptr %ptr
  %val7 = load volatile double, ptr %ptr
  %val8 = load volatile double, ptr %ptr
  %val9 = load volatile double, ptr %ptr
  %val10 = load volatile double, ptr %ptr
  %val11 = load volatile double, ptr %ptr
  %val12 = load volatile double, ptr %ptr
  %val13 = load volatile double, ptr %ptr
  %val14 = load volatile double, ptr %ptr
  %val15 = load volatile double, ptr %ptr
  %val16 = load volatile double, ptr %ptr

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

  store volatile double %val0, ptr %ptr
  store volatile double %val1, ptr %ptr
  store volatile double %val2, ptr %ptr
  store volatile double %val3, ptr %ptr
  store volatile double %val4, ptr %ptr
  store volatile double %val5, ptr %ptr
  store volatile double %val6, ptr %ptr
  store volatile double %val7, ptr %ptr
  store volatile double %val8, ptr %ptr
  store volatile double %val9, ptr %ptr
  store volatile double %val10, ptr %ptr
  store volatile double %val11, ptr %ptr
  store volatile double %val12, ptr %ptr
  store volatile double %val13, ptr %ptr
  store volatile double %val14, ptr %ptr
  store volatile double %val15, ptr %ptr
  store volatile double %val16, ptr %ptr

  store volatile double %sqrt0, ptr %ptr
  store volatile double %sqrt1, ptr %ptr
  store volatile double %sqrt2, ptr %ptr
  store volatile double %sqrt3, ptr %ptr
  store volatile double %sqrt4, ptr %ptr
  store volatile double %sqrt5, ptr %ptr
  store volatile double %sqrt6, ptr %ptr
  store volatile double %sqrt7, ptr %ptr
  store volatile double %sqrt8, ptr %ptr
  store volatile double %sqrt9, ptr %ptr
  store volatile double %sqrt10, ptr %ptr
  store volatile double %sqrt11, ptr %ptr
  store volatile double %sqrt12, ptr %ptr
  store volatile double %sqrt13, ptr %ptr
  store volatile double %sqrt14, ptr %ptr
  store volatile double %sqrt15, ptr %ptr
  store volatile double %sqrt16, ptr %ptr

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
