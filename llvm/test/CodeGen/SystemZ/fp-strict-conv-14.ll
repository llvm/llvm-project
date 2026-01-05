; Test strict conversion of floating-point values to unsigned integers.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

declare i32 @llvm.experimental.constrained.fptoui.i32.f16(half, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f64(double, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128, metadata)

declare i64 @llvm.experimental.constrained.fptoui.i64.f16(half, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f32(float, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f64(double, metadata)
declare i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128, metadata)

; Test f16->i32.
define i32 @f0(half %f) #0 {
; CHECK-LABEL: f0:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT: clfebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f16(half %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f32->i32.
define i32 @f1(float %f) #0 {
; CHECK-LABEL: f1:
; CHECK: clfebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f64->i32.
define i32 @f2(double %f) #0 {
; CHECK-LABEL: f2:
; CHECK: clfdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f128->i32.
define i32 @f3(ptr %src) #0 {
; CHECK-LABEL: f3:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clfxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = call i32 @llvm.experimental.constrained.fptoui.i32.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i32 %conv
}

; Test f16->i64.
define i64 @f4(half %f) #0 {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, __extendhfsf2@PLT
; CHECK-NEXT: clgebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f16(half %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f32->i64.
define i64 @f5(float %f) #0 {
; CHECK-LABEL: f5:
; CHECK: clgebr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f32(float %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f64->i64.
define i64 @f6(double %f) #0 {
; CHECK-LABEL: f6:
; CHECK: clgdbr %r2, 5, %f0, 0
; CHECK: br %r14
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f64(double %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

; Test f128->i64.
define i64 @f7(ptr %src) #0 {
; CHECK-LABEL: f7:
; CHECK-DAG: ld %f0, 0(%r2)
; CHECK-DAG: ld %f2, 8(%r2)
; CHECK: clgxbr %r2, 5, %f0, 0
; CHECK: br %r14
  %f = load fp128, ptr %src
  %conv = call i64 @llvm.experimental.constrained.fptoui.i64.f128(fp128 %f,
                                               metadata !"fpexcept.strict") #0
  ret i64 %conv
}

attributes #0 = { strictfp }
