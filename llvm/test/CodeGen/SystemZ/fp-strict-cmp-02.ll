; Test 64-bit floating-point strict comparison.  The tests assume a z10
; implementation of select, using conditional branches rather than LOCGR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -verify-machineinstrs\
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-VECTOR %s

declare double @foo()

; Check comparison with registers.
define i64 @f1(i64 %a, i64 %b, double %f1, double %f2) #0 {
; CHECK-LABEL: f1:
; CHECK: cdbr %f0, %f2
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the low end of the CDB range.
define i64 @f2(i64 %a, i64 %b, double %f1, double *%ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: cdb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f2 = load double, double *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the high end of the aligned CDB range.
define i64 @f3(i64 %a, i64 %b, double %f1, double *%base) #0 {
; CHECK-LABEL: f3:
; CHECK: cdb %f0, 4088(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %f2 = load double, double *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f4(i64 %a, i64 %b, double %f1, double *%base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r4, 4096
; CHECK: cdb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %f2 = load double, double *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check negative displacements, which also need separate address logic.
define i64 @f5(i64 %a, i64 %b, double %f1, double *%base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r4, -8
; CHECK: cdb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %f2 = load double, double *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that CDB allows indices.
define i64 @f6(i64 %a, i64 %b, double %f1, double *%base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r5, 3
; CHECK: cdb %f0, 800(%r1,%r4)
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %f2 = load double, double *%ptr2
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check that comparisons of spilled values can use CDB rather than CDBR.
define double @f7(double *%ptr0) #0 {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK-SCALAR: cdb {{%f[0-9]+}}, 160(%r15)
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

  %cmp0 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp1 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val1,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp2 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val2,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp3 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val3,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp4 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val4,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp5 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val5,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp6 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val6,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp7 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val7,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp8 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val8,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp9 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val9,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  %cmp10 = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %ret, double %val10,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0

  %sel0 = select i1 %cmp0, double %ret, double 0.0
  %sel1 = select i1 %cmp1, double %sel0, double 1.0
  %sel2 = select i1 %cmp2, double %sel1, double 2.0
  %sel3 = select i1 %cmp3, double %sel2, double 3.0
  %sel4 = select i1 %cmp4, double %sel3, double 4.0
  %sel5 = select i1 %cmp5, double %sel4, double 5.0
  %sel6 = select i1 %cmp6, double %sel5, double 6.0
  %sel7 = select i1 %cmp7, double %sel6, double 7.0
  %sel8 = select i1 %cmp8, double %sel7, double 8.0
  %sel9 = select i1 %cmp9, double %sel8, double 9.0
  %sel10 = select i1 %cmp10, double %sel9, double 10.0

  ret double %sel10
}

; Check comparison with zero.
define i64 @f8(i64 %a, i64 %b, double %f) #0 {
; CHECK-LABEL: f8:
; CHECK-SCALAR: ltdbr %f0, %f0
; CHECK-SCALAR-NEXT: ber %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR: ltdbr %f0, %f0
; CHECK-VECTOR-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f, double 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

; Check the comparison can be reversed if that allows CDB to be used,
define i64 @f9(i64 %a, i64 %b, double %f2, double *%ptr) #0 {
; CHECK-LABEL: f9:
; CHECK: cdb %f0, 0(%r4)
; CHECK-SCALAR-NEXT: blr %r14
; CHECK-SCALAR: lgr %r2, %r3
; CHECK-VECTOR-NEXT: locgrnl %r2, %r3
; CHECK: br %r14
  %f1 = load double, double *%ptr
  %cond = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %f1, double %f2,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

attributes #0 = { strictfp }

declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)

