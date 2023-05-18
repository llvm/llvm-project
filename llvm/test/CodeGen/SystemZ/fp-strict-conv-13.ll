; Test strict conversions of unsigned integers to floating-point values
; (z196 and above).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32, metadata, metadata)

declare float @llvm.experimental.constrained.uitofp.f32.i64(i64, metadata, metadata)
declare double @llvm.experimental.constrained.uitofp.f64.i64(i64, metadata, metadata)
declare fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64, metadata, metadata)

; Check i32->f32.
define float @f1(i32 %i) #0 {
; CHECK-LABEL: f1:
; CHECK: celfbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = call float @llvm.experimental.constrained.uitofp.f32.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) #0 {
; CHECK-LABEL: f2:
; CHECK: cdlfbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = call double @llvm.experimental.constrained.uitofp.f64.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) #0 {
; CHECK-LABEL: f3:
; CHECK: cxlfbr %f0, 0, %r2, 0
; CHECK-DAG: std %f0, 0(%r3)
; CHECK-DAG: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.uitofp.f128.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, ptr %dst
  ret void
}

; Check i64->f32.
define float @f4(i64 %i) #0 {
; CHECK-LABEL: f4:
; CHECK: celgbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = call float @llvm.experimental.constrained.uitofp.f32.i64(i64 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %conv
}

; Check i64->f64.
define double @f5(i64 %i) #0 {
; CHECK-LABEL: f5:
; CHECK: cdlgbr %f0, 0, %r2, 0
; CHECK: br %r14
  %conv = call double @llvm.experimental.constrained.uitofp.f64.i64(i64 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %conv
}

; Check i64->f128.
define void @f6(i64 %i, ptr %dst) #0 {
; CHECK-LABEL: f6:
; CHECK: cxlgbr %f0, 0, %r2, 0
; CHECK-DAG: std %f0, 0(%r3)
; CHECK-DAG: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.uitofp.f128.i64(i64 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, ptr %dst
  ret void
}

attributes #0 = { strictfp }
