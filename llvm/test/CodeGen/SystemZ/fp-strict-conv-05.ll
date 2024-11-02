; Test strict conversions of signed i32s to floating-point values.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
declare fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32, metadata, metadata)

; Check i32->f32.
define float @f1(i32 %i) #0 {
; CHECK-LABEL: f1:
; CHECK: cefbr %f0, %r2
; CHECK: br %r14
  %conv = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %conv
}

; Check i32->f64.
define double @f2(i32 %i) #0 {
; CHECK-LABEL: f2:
; CHECK: cdfbr %f0, %r2
; CHECK: br %r14
  %conv = call double @llvm.experimental.constrained.sitofp.f64.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret double %conv
}

; Check i32->f128.
define void @f3(i32 %i, ptr %dst) #0 {
; CHECK-LABEL: f3:
; CHECK: cxfbr %f0, %r2
; CHECK: std %f0, 0(%r3)
; CHECK: std %f2, 8(%r3)
; CHECK: br %r14
  %conv = call fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32 %i,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store fp128 %conv, ptr %dst
  ret void
}

attributes #0 = { strictfp }
