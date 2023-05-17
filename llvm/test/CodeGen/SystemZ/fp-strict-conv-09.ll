; Test strict conversion of floating-point values to signed i32s.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i32 @llvm.experimental.constrained.fptosi.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128, metadata)

; Test f32->i32.
define i32 @f1(float %f) #0 {
; CHECK-LABEL: f1:
; CHECK: cfebr %r2, 5, %f0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptosi.i32.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) #0 {
; CHECK-LABEL: f2:
; CHECK: cfdbr %r2, 5, %f0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptosi.i32.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(ptr %src) #0 {
; CHECK-LABEL: f3:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: cfxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = call i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

attributes #0 = { strictfp }
