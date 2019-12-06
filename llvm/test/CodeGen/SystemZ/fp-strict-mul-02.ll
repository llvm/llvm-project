; Test strict multiplication of two f32s, producing an f64 result.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @foo()
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)
declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)

; Check register multiplication.
define double @f1(float %f1, float %f2) #0 {
; CHECK-LABEL: f1:
; CHECK: mdebr %f0, %f2
; CHECK: br %r14
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the low end of the MDEB range.
define double @f2(float %f1, float *%ptr) #0 {
; CHECK-LABEL: f2:
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float, float *%ptr
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the high end of the aligned MDEB range.
define double @f3(float %f1, float *%base) #0 {
; CHECK-LABEL: f3:
; CHECK: mdeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %f2 = load float, float *%ptr
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(float %f1, float *%base) #0 {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %f2 = load float, float *%ptr
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(float %f1, float *%base) #0 {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %f2 = load float, float *%ptr
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that MDEB allows indices.
define double @f6(float %f1, float *%base, i64 %index) #0 {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: mdeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %f2 = load float, float *%ptr2
  %f1x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f1,
                        metadata !"fpexcept.strict") #0
  %f2x = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %f2,
                        metadata !"fpexcept.strict") #0
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  ret double %res
}

; Check that multiplications of spilled values can use MDEB rather than MDEBR.
define float @f7(float *%ptr0) #0 {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, foo@PLT
; CHECK: mdeb %f0, 16{{[04]}}(%r15)
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

  %val0 = load float, float *%ptr0
  %val1 = load float, float *%ptr1
  %val2 = load float, float *%ptr2
  %val3 = load float, float *%ptr3
  %val4 = load float, float *%ptr4
  %val5 = load float, float *%ptr5
  %val6 = load float, float *%ptr6
  %val7 = load float, float *%ptr7
  %val8 = load float, float *%ptr8
  %val9 = load float, float *%ptr9
  %val10 = load float, float *%ptr10

  %frob0 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val0, float %val0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob1 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val1, float %val1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob2 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val2, float %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob3 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val3, float %val3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob4 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val4, float %val4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob5 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val5, float %val5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob6 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val6, float %val6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob7 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val7, float %val7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob8 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val8, float %val8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob9 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val9, float %val9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %frob10 = call float @llvm.experimental.constrained.fadd.f32(
                        float %val10, float %val10,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  store float %frob0, float *%ptr0
  store float %frob1, float *%ptr1
  store float %frob2, float *%ptr2
  store float %frob3, float *%ptr3
  store float %frob4, float *%ptr4
  store float %frob5, float *%ptr5
  store float %frob6, float *%ptr6
  store float %frob7, float *%ptr7
  store float %frob8, float *%ptr8
  store float %frob9, float *%ptr9
  store float %frob10, float *%ptr10

  %ret = call float @foo() #0

  %accext0 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %ret,
                        metadata !"fpexcept.strict") #0
  %ext0 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob0,
                        metadata !"fpexcept.strict") #0
  %mul0 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext0, double %ext0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra0 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul0, double 1.01,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc0 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext1 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc0,
                        metadata !"fpexcept.strict") #0
  %ext1 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob1,
                        metadata !"fpexcept.strict") #0
  %mul1 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext1, double %ext1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra1 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul1, double 1.11,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc1 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext2 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc1,
                        metadata !"fpexcept.strict") #0
  %ext2 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob2,
                        metadata !"fpexcept.strict") #0
  %mul2 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext2, double %ext2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra2 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul2, double 1.21,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc2 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext3 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc2,
                        metadata !"fpexcept.strict") #0
  %ext3 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob3,
                        metadata !"fpexcept.strict") #0
  %mul3 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext3, double %ext3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra3 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul3, double 1.31,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc3 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext4 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc3,
                        metadata !"fpexcept.strict") #0
  %ext4 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob4,
                        metadata !"fpexcept.strict") #0
  %mul4 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext4, double %ext4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra4 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul4, double 1.41,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc4 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext5 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc4,
                        metadata !"fpexcept.strict") #0
  %ext5 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob5,
                        metadata !"fpexcept.strict") #0
  %mul5 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext5, double %ext5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra5 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul5, double 1.51,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc5 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext6 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc5,
                        metadata !"fpexcept.strict") #0
  %ext6 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob6,
                        metadata !"fpexcept.strict") #0
  %mul6 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext6, double %ext6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra6 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul6, double 1.61,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc6 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext7 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc6,
                        metadata !"fpexcept.strict") #0
  %ext7 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob7,
                        metadata !"fpexcept.strict") #0
  %mul7 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext7, double %ext7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra7 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul7, double 1.71,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc7 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext8 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc7,
                        metadata !"fpexcept.strict") #0
  %ext8 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob8,
                        metadata !"fpexcept.strict") #0
  %mul8 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext8, double %ext8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra8 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul8, double 1.81,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc8 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  %accext9 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %trunc8,
                        metadata !"fpexcept.strict") #0
  %ext9 = call double @llvm.experimental.constrained.fpext.f64.f32(
                        float %frob9,
                        metadata !"fpexcept.strict") #0
  %mul9 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext9, double %ext9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %extra9 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul9, double 1.91,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %trunc9 = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                        double %extra9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0

  ret float %trunc9
}

attributes #0 = { strictfp }
