; Test strict conversion of floating-point values to signed i64s.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @llvm.experimental.constrained.fptosi.i64.f16(half, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f32(float, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128, metadata)

; Test f16->i64.
define i64 @f0(half %f) #0 {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT: cgebr %r2, 5, %f0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptosi.i64.f16(half %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f32->i64.
define i64 @f1(float %f) #0 {
; CHECK-LABEL: f1:
; CHECK: cgebr %r2, 5, %f0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptosi.i64.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f64->i64.
define i64 @f2(double %f) #0 {
; CHECK-LABEL: f2:
; CHECK: cgdbr %r2, 5, %f0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptosi.i64.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f128->i64.
define i64 @f3(ptr %src) #0 {
; CHECK-LABEL: f3:
; CHECK: ld %f0, 0(%r2)
; CHECK: ld %f2, 8(%r2)
; CHECK: cgxbr %r2, 5, %f0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = call i64 @llvm.experimental.constrained.fptosi.i64.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

attributes #0 = { strictfp }
