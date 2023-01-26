; Test strict conversions of unsigned i32s to floating-point values (z10 only).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32, metadata, metadata)

; Check i32->f32.  There is no native instruction, so we must promote
; to i64 first.
define float @f1(i32 %i) #0 {
; CHECK-LABEL: f1:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cegbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) #0 {
; CHECK-LABEL: f2:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cdgbr %f0, [[REGISTER]]
; CHECK: br %r14
  %conv = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) #0 {
; CHECK-LABEL: f3:
; CHECK: llgfr [[REGISTER:%r[0-5]]], %r2
; CHECK: cxgbr %f0, [[REGISTER]]
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, ptr %dst
  ret void
}

attributes #0 = { strictfp }
