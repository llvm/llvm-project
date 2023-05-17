; Test strict 64-bit floating-point division.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare double @foo()
declare double @llvm.experimental.constrained.fdiv.f64(double, double, metadata, metadata)

; Check register division.
define double @f1(double %f1, double %f2) #0 {
; CHECK-LABEL: f1:
; CHECK: ddbr %f0, %f2
; CHECK: br %r14
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the low end of the DDB range.
define double @f2(double %f1, ptr %ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load double, ptr %ptr
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the high end of the aligned DDB range.
define double @f3(double %f1, ptr %base) #0 {
; CHECK-LABEL: f3:
; CHECK: ddb %f0, 4088(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 511
  %f2 = load double, ptr %ptr
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(double %f1, ptr %base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 512
  %f2 = load double, ptr %ptr
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(double %f1, ptr %base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: ddb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr double, ptr %base, i64 -1
  %f2 = load double, ptr %ptr
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that DDB allows indices.
define double @f6(double %f1, ptr %base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: ddb %f0, 800(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %base, i64 %index
  %ptr2 = getelementptr double, ptr %ptr1, i64 100
  %f2 = load double, ptr %ptr2
  %res = call double @llvm.experimental.constrained.fdiv.f64(
                        double %f1, double %f2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that divisions of spilled values can use DDB rather than DDBR.
define double @f7(ptr %ptr0) #0 {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: ddb %f0, 160(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr double, ptr %ptr0, i64 2
  %ptr2 = getelementptr double, ptr %ptr0, i64 4
  %ptr3 = getelementptr double, ptr %ptr0, i64 6
  %ptr4 = getelementptr double, ptr %ptr0, i64 8
  %ptr5 = getelementptr double, ptr %ptr0, i64 10
  %ptr6 = getelementptr double, ptr %ptr0, i64 12
  %ptr7 = getelementptr double, ptr %ptr0, i64 14
  %ptr8 = getelementptr double, ptr %ptr0, i64 16
  %ptr9 = getelementptr double, ptr %ptr0, i64 18
  %ptr10 = getelementptr double, ptr %ptr0, i64 20

  %val0 = load double, ptr %ptr0
  %val1 = load double, ptr %ptr1
  %val2 = load double, ptr %ptr2
  %val3 = load double, ptr %ptr3
  %val4 = load double, ptr %ptr4
  %val5 = load double, ptr %ptr5
  %val6 = load double, ptr %ptr6
  %val7 = load double, ptr %ptr7
  %val8 = load double, ptr %ptr8
  %val9 = load double, ptr %ptr9
  %val10 = load double, ptr %ptr10

  %ret = call double @foo() #0

  %div0 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %ret, double %val0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div1 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div0, double %val1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div2 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div1, double %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div3 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div2, double %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div4 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div3, double %val4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div5 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div4, double %val5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div6 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div5, double %val6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div7 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div6, double %val7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div8 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div7, double %val8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div9 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div8, double %val9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %div10 = call double @llvm.experimental.constrained.fdiv.f64(
                        double %div9, double %val10,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  ret double %div10
}

attributes #0 = { strictfp }
