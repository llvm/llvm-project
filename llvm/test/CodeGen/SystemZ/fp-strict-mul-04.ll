; Test strict multiplication of two f64s, producing an f128 result.
; FIXME: we do not have a strict version of fpext yet
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare fp128 @llvm.experimental.constrained.fmul.f128(fp128, fp128, metadata, metadata)

declare double @foo()

; Check register multiplication.  "mxdbr %f0, %f2" is not valid from LLVM's
; point of view, because %f2 is the low register of the FP128 %f0.  Pass the
; multiplier in %f4 instead.
define void @f1(double %f1, double %dummy, double %f2, fp128 *%dst) {
; CHECK-LABEL: f1:
; CHECK: mxdbr %f0, %f4
; CHECK: std %f0, 0(%r2)
; CHECK: std %f2, 8(%r2)
; CHECK: br %r14
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the low end of the MXDB range.
define void @f2(double %f1, double *%ptr, fp128 *%dst) {
; CHECK-LABEL: f2:
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %f2 = load double, double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the high end of the aligned MXDB range.
define void @f3(double %f1, double *%base, fp128 *%dst) {
; CHECK-LABEL: f3:
; CHECK: mxdb %f0, 4088(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 511
  %f2 = load double, double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check the next doubleword up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f4(double %f1, double *%base, fp128 *%dst) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 512
  %f2 = load double, double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check negative displacements, which also need separate address logic.
define void @f5(double %f1, double *%base, fp128 *%dst) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -8
; CHECK: mxdb %f0, 0(%r2)
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %ptr = getelementptr double, double *%base, i64 -1
  %f2 = load double, double *%ptr
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that MXDB allows indices.
define void @f6(double %f1, double *%base, i64 %index, fp128 *%dst) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 3
; CHECK: mxdb %f0, 800(%r1,%r2)
; CHECK: std %f0, 0(%r4)
; CHECK: std %f2, 8(%r4)
; CHECK: br %r14
  %ptr1 = getelementptr double, double *%base, i64 %index
  %ptr2 = getelementptr double, double *%ptr1, i64 100
  %f2 = load double, double *%ptr2
  %f1x = fpext double %f1 to fp128
  %f2x = fpext double %f2 to fp128
  %res = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %f1x, fp128 %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store fp128 %res, fp128 *%dst
  ret void
}

; Check that multiplications of spilled values can use MXDB rather than MXDBR.
define double @f7(double *%ptr0) {
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

  %frob0 = fadd double %val0, %val0
  %frob1 = fadd double %val1, %val1
  %frob2 = fadd double %val2, %val2
  %frob3 = fadd double %val3, %val3
  %frob4 = fadd double %val4, %val4
  %frob5 = fadd double %val5, %val5
  %frob6 = fadd double %val6, %val6
  %frob7 = fadd double %val7, %val7
  %frob8 = fadd double %val8, %val8
  %frob9 = fadd double %val9, %val9
  %frob10 = fadd double %val9, %val10

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

  %ret = call double @foo()

  %accext0 = fpext double %ret to fp128
  %ext0 = fpext double %frob0 to fp128
  %mul0 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext0, fp128 %ext0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const0 = fpext double 1.01 to fp128
  %extra0 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul0, fp128 %const0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc0 = fptrunc fp128 %extra0 to double

  %accext1 = fpext double %trunc0 to fp128
  %ext1 = fpext double %frob1 to fp128
  %mul1 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext1, fp128 %ext1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const1 = fpext double 1.11 to fp128
  %extra1 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul1, fp128 %const1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc1 = fptrunc fp128 %extra1 to double

  %accext2 = fpext double %trunc1 to fp128
  %ext2 = fpext double %frob2 to fp128
  %mul2 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext2, fp128 %ext2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const2 = fpext double 1.21 to fp128
  %extra2 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul2, fp128 %const2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc2 = fptrunc fp128 %extra2 to double

  %accext3 = fpext double %trunc2 to fp128
  %ext3 = fpext double %frob3 to fp128
  %mul3 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext3, fp128 %ext3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const3 = fpext double 1.31 to fp128
  %extra3 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul3, fp128 %const3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc3 = fptrunc fp128 %extra3 to double

  %accext4 = fpext double %trunc3 to fp128
  %ext4 = fpext double %frob4 to fp128
  %mul4 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext4, fp128 %ext4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const4 = fpext double 1.41 to fp128
  %extra4 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul4, fp128 %const4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc4 = fptrunc fp128 %extra4 to double

  %accext5 = fpext double %trunc4 to fp128
  %ext5 = fpext double %frob5 to fp128
  %mul5 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext5, fp128 %ext5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const5 = fpext double 1.51 to fp128
  %extra5 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul5, fp128 %const5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc5 = fptrunc fp128 %extra5 to double

  %accext6 = fpext double %trunc5 to fp128
  %ext6 = fpext double %frob6 to fp128
  %mul6 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext6, fp128 %ext6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const6 = fpext double 1.61 to fp128
  %extra6 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul6, fp128 %const6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc6 = fptrunc fp128 %extra6 to double

  %accext7 = fpext double %trunc6 to fp128
  %ext7 = fpext double %frob7 to fp128
  %mul7 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext7, fp128 %ext7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const7 = fpext double 1.71 to fp128
  %extra7 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul7, fp128 %const7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc7 = fptrunc fp128 %extra7 to double

  %accext8 = fpext double %trunc7 to fp128
  %ext8 = fpext double %frob8 to fp128
  %mul8 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext8, fp128 %ext8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const8 = fpext double 1.81 to fp128
  %extra8 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul8, fp128 %const8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc8 = fptrunc fp128 %extra8 to double

  %accext9 = fpext double %trunc8 to fp128
  %ext9 = fpext double %frob9 to fp128
  %mul9 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %accext9, fp128 %ext9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %const9 = fpext double 1.91 to fp128
  %extra9 = call fp128 @llvm.experimental.constrained.fmul.f128(
                        fp128 %mul9, fp128 %const9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc9 = fptrunc fp128 %extra9 to double

  ret double %trunc9
}
