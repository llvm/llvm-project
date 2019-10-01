; Test strict multiplication of two f32s, producing an f64 result.
; FIXME: we do not have a strict version of fpext yet
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @foo()
declare double @llvm.experimental.constrained.fmul.f64(double, double, metadata, metadata)

; Check register multiplication.
define double @f1(float %f1, float %f2) {
; CHECK-LABEL: f1:
; CHECK: mdebr %f0, %f2
; CHECK: br %r14
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check the low end of the MDEB range.
define double @f2(float %f1, float *%ptr) {
; CHECK-LABEL: f2:
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %f2 = load float, float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check the high end of the aligned MDEB range.
define double @f3(float %f1, float *%base) {
; CHECK-LABEL: f3:
; CHECK: mdeb %f0, 4092(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1023
  %f2 = load float, float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f4(float %f1, float *%base) {
; CHECK-LABEL: f4:
; CHECK: aghi %r2, 4096
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 1024
  %f2 = load float, float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check negative displacements, which also need separate address logic.
define double @f5(float %f1, float *%base) {
; CHECK-LABEL: f5:
; CHECK: aghi %r2, -4
; CHECK: mdeb %f0, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr float, float *%base, i64 -1
  %f2 = load float, float *%ptr
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check that MDEB allows indices.
define double @f6(float %f1, float *%base, i64 %index) {
; CHECK-LABEL: f6:
; CHECK: sllg %r1, %r3, 2
; CHECK: mdeb %f0, 400(%r1,%r2)
; CHECK: br %r14
  %ptr1 = getelementptr float, float *%base, i64 %index
  %ptr2 = getelementptr float, float *%ptr1, i64 100
  %f2 = load float, float *%ptr2
  %f1x = fpext float %f1 to double
  %f2x = fpext float %f2 to double
  %res = call double @llvm.experimental.constrained.fmul.f64(
                        double %f1x, double %f2x,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret double %res
}

; Check that multiplications of spilled values can use MDEB rather than MDEBR.
define float @f7(float *%ptr0) {
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

  %frob0 = fadd float %val0, %val0
  %frob1 = fadd float %val1, %val1
  %frob2 = fadd float %val2, %val2
  %frob3 = fadd float %val3, %val3
  %frob4 = fadd float %val4, %val4
  %frob5 = fadd float %val5, %val5
  %frob6 = fadd float %val6, %val6
  %frob7 = fadd float %val7, %val7
  %frob8 = fadd float %val8, %val8
  %frob9 = fadd float %val9, %val9
  %frob10 = fadd float %val9, %val10

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

  %ret = call float @foo()

  %accext0 = fpext float %ret to double
  %ext0 = fpext float %frob0 to double
  %mul0 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext0, double %ext0,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra0 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul0, double 1.01,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc0 = fptrunc double %extra0 to float

  %accext1 = fpext float %trunc0 to double
  %ext1 = fpext float %frob1 to double
  %mul1 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext1, double %ext1,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra1 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul1, double 1.11,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc1 = fptrunc double %extra1 to float

  %accext2 = fpext float %trunc1 to double
  %ext2 = fpext float %frob2 to double
  %mul2 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext2, double %ext2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra2 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul2, double 1.21,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc2 = fptrunc double %extra2 to float

  %accext3 = fpext float %trunc2 to double
  %ext3 = fpext float %frob3 to double
  %mul3 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext3, double %ext3,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra3 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul3, double 1.31,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc3 = fptrunc double %extra3 to float

  %accext4 = fpext float %trunc3 to double
  %ext4 = fpext float %frob4 to double
  %mul4 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext4, double %ext4,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra4 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul4, double 1.41,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc4 = fptrunc double %extra4 to float

  %accext5 = fpext float %trunc4 to double
  %ext5 = fpext float %frob5 to double
  %mul5 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext5, double %ext5,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra5 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul5, double 1.51,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc5 = fptrunc double %extra5 to float

  %accext6 = fpext float %trunc5 to double
  %ext6 = fpext float %frob6 to double
  %mul6 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext6, double %ext6,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra6 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul6, double 1.61,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc6 = fptrunc double %extra6 to float

  %accext7 = fpext float %trunc6 to double
  %ext7 = fpext float %frob7 to double
  %mul7 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext7, double %ext7,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra7 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul7, double 1.71,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc7 = fptrunc double %extra7 to float

  %accext8 = fpext float %trunc7 to double
  %ext8 = fpext float %frob8 to double
  %mul8 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext8, double %ext8,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra8 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul8, double 1.81,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc8 = fptrunc double %extra8 to float

  %accext9 = fpext float %trunc8 to double
  %ext9 = fpext float %frob9 to double
  %mul9 = call double @llvm.experimental.constrained.fmul.f64(
                        double %accext9, double %ext9,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %extra9 = call double @llvm.experimental.constrained.fmul.f64(
                        double %mul9, double 1.91,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  %trunc9 = fptrunc double %extra9 to float

  ret float %trunc9
}
