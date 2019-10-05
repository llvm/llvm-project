; Test strict 64-bit floating-point addition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs | FileCheck %s
declare double @foo()
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

; Check register addition.
define double @f1(double %f1, double %f2) #0 {
; CHECK-LABEL: f1:
; CHECK: adbr %f0, %f2
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the low end of the ADB range.
define double @f2(double %f1, double *%ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double, double *%ptr
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the high end of the aligned ADB range.
define double @f3(double %f1, double *%base) #0 {
; CHECK-LABEL: f3:
; CHECK: adb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %f2 = load double, double *%ptr
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, double *%base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %f2 = load double, double *%ptr
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, double *%base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: adb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %f2 = load double, double *%ptr
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that ADB allows indices.
define double @f6(double %f1, double *%base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: adb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %f2 = load double, double *%ptr2
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that additions of spilled values can use ADB rather than ADBR.
define double @f7(double *%ptr0) #0 {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: adb %f0, 160(%r15)
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

  %ret = call double @foo() #0

  %add0 = call double @llvm.experimental.constrained.fadd.f64(
                        double %ret, double %val0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add1 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add0, double %val1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add2 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add1, double %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add3 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add2, double %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add4 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add3, double %val4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add5 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add4, double %val5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add6 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add5, double %val6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add7 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add6, double %val7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add8 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add7, double %val8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add9 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add8, double %val9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %add10 = call double @llvm.experimental.constrained.fadd.f64(
                        double %add9, double %val10,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  ret double %add10
}

attributes #0 = { strictfp }
