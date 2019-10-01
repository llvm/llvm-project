; Test strict floating-point truncations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-SCALAR %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 \
; RUN:   | FileCheck -check-prefix=CHECK -check-prefix=CHECK-VECTOR %s

declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f128(fp128, metadata, metadata)
declare double @llvm.experimental.constrained.fptrunc.f64.f128(fp128, metadata, metadata)

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)

; Test f64->f32.
define float @f1(double %d1, double %d2) {
; CHECK-LABEL: f1:
; CHECK-SCALAR: ledbr %f0, %f2
; CHECK-VECTOR: ledbra %f0, 0, %f2, 0
; CHECK: br %r14
  %res = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                                               double %d2,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret float %res
}

; Test f128->f32.
define float @f2(fp128 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: lexbr %f0, %f0
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %res = call float @llvm.experimental.constrained.fptrunc.f32.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret float %res
}

; Make sure that we don't use %f0 as the destination of LEXBR when %f2
; is still live.
define void @f3(float *%dst, fp128 *%ptr, float %d1, float %d2) {
; CHECK-LABEL: f3:
; CHECK: lexbr %f1, %f1
; CHECK: aebr %f1, %f2
; CHECK: ste %f1, 0(%r2)
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %conv = call float @llvm.experimental.constrained.fptrunc.f32.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %conv, float %d2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store float %res, float *%dst
  ret void
}

; Test f128->f64.
define double @f4(fp128 *%ptr) {
; CHECK-LABEL: f4:
; CHECK: ldxbr %f0, %f0
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %res = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  ret double %res
}

; Like f3, but for f128->f64.
define void @f5(double *%dst, fp128 *%ptr, double %d1, double %d2) {
; CHECK-LABEL: f5:
; CHECK: ldxbr %f1, %f1
; CHECK-SCALAR: adbr %f1, %f2
; CHECK-SCALAR: std %f1, 0(%r2)
; CHECK-VECTOR: wfadb [[REG:%f[0-9]+]], %f1, %f2
; CHECK-VECTOR: std [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load fp128, fp128 *%ptr
  %conv = call double @llvm.experimental.constrained.fptrunc.f64.f128(
                                               fp128 %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict")
  %res = call double @llvm.experimental.constrained.fadd.f64(
                        double %conv, double %d2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  store double %res, double *%dst
  ret void
}
