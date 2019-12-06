; Test strict multiplication of two f64s, producing an f128 result.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fmul.f128(fp128, fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f128(fp128, metadata, metadata)
declare fp128 @llvm.experimental.constrained.fpext.f128.f64(double, metadata)

declare double @foo()

; Check register multiplication.  "mxdbr %f0, %f2" is not valid from LLVM's
; point of view, because %f2 is the low register of the FP128 %f0.  Pass the
; multiplier in %f4 instead.
define void @f1(double %f1, double %dummy, double %f2, fp128 *%dst) #0 {
; CHECK-LABEL: f1:
; CHECK: mxdbr %f0, %f4
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the low end of the MXDB range.
define void @f2(double %f1, double *%ptr, fp128 *%dst) #0 {
; CHECK-LABEL: f2:
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %f2 = load double, double *%ptr
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the high end of the aligned MXDB range.
define void @f3(double %f1, double *%base, fp128 *%dst) #0 {
; CHECK-LABEL: f3:
; CHECK: mxdb %f0, 4088(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %f2 = load double, double *%ptr
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(double %f1, double *%base, fp128 *%dst) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %f2 = load double, double *%ptr
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(double %f1, double *%base, fp128 *%dst) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %f2 = load double, double *%ptr
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that MXDB allows indices.
define void @f6(double %f1, double *%base, i64 %index, fp128 *%dst) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: mxdb %f0, 800(%r1,%r2)
; CHECK: std %f0, 0(%r4)
; CHECK: std %f2, 8(%r4)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %f2 = load double, double *%ptr2
  %f1x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %f2,
                        metadata !"fpexcept.strict") #0
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that multiplications of spilled values can use MXDB rather than MXDBR.
define double @f7(double *%ptr0) #0 {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: mxdb %f0, 160(%r15)
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

  %frob0 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val0, double %val0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob1 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val1, double %val1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob2 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val2, double %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob3 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val3, double %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob4 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val4, double %val4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob5 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val5, double %val5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob6 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val6, double %val6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob7 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val7, double %val7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob8 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val8, double %val8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob9 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val9, double %val9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob10 = call double @llvm.experimental.constrained.fadd.f64(
                        double %val10, double %val10,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  store double %frob0, double *%ptr0
  store double %frob1, double *%ptr1
  store double %frob2, double *%ptr2
  store double %frob3, double *%ptr3
  store double %frob4, double *%ptr4
  store double %frob5, double *%ptr5
  store double %frob6, double *%ptr6
  store double %frob7, double *%ptr7
  store double %frob8, double *%ptr8
  store double %frob9, double *%ptr9
  store double %frob10, double *%ptr10

  %ret = call double @foo() #0

  %accext0 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %ret,
                        metadata !"fpexcept.strict") #0
  %ext0 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob0,
                        metadata !"fpexcept.strict") #0
  %mul0 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext0, fp128 %ext0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra0 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul0, fp128 0xL00000000000000003fff000001000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc0 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext1 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc0,
                        metadata !"fpexcept.strict") #0
  %ext1 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob1,
                        metadata !"fpexcept.strict") #0
  %mul1 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext1, fp128 %ext1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra1 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul1, fp128 0xL00000000000000003fff000002000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc1 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext2 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc1,
                        metadata !"fpexcept.strict") #0
  %ext2 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob2,
                        metadata !"fpexcept.strict") #0
  %mul2 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext2, fp128 %ext2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra2 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul2, fp128 0xL00000000000000003fff000003000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc2 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext3 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc2,
                        metadata !"fpexcept.strict") #0
  %ext3 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob3,
                        metadata !"fpexcept.strict") #0
  %mul3 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext3, fp128 %ext3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra3 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul3, fp128 0xL00000000000000003fff000004000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc3 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext4 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc3,
                        metadata !"fpexcept.strict") #0
  %ext4 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob4,
                        metadata !"fpexcept.strict") #0
  %mul4 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext4, fp128 %ext4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra4 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul4, fp128 0xL00000000000000003fff000005000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc4 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext5 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc4,
                        metadata !"fpexcept.strict") #0
  %ext5 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob5,
                        metadata !"fpexcept.strict") #0
  %mul5 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext5, fp128 %ext5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra5 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul5, fp128 0xL00000000000000003fff000006000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc5 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext6 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc5,
                        metadata !"fpexcept.strict") #0
  %ext6 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob6,
                        metadata !"fpexcept.strict") #0
  %mul6 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext6, fp128 %ext6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra6 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul6, fp128 0xL00000000000000003fff000007000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc6 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext7 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc6,
                        metadata !"fpexcept.strict") #0
  %ext7 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob7,
                        metadata !"fpexcept.strict") #0
  %mul7 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext7, fp128 %ext7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra7 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul7, fp128 0xL00000000000000003fff000008000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc7 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext8 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc7,
                        metadata !"fpexcept.strict") #0
  %ext8 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob8,
                        metadata !"fpexcept.strict") #0
  %mul8 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext8, fp128 %ext8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra8 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul8, fp128 0xL00000000000000003fff000009000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc8 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext9 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %trunc8,
                        metadata !"fpexcept.strict") #0
  %ext9 = call fp128 @llvm.experimental.constrained.fpext.f128.f64(
                        double %frob9,
                        metadata !"fpexcept.strict") #0
  %mul9 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext9, fp128 %ext9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra9 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul9, fp128 0xL00000000000000003fff00000a000000,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc9 = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                        fp128 %extra9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  ret double %trunc9
}

attributes #0 = { strictfp }
