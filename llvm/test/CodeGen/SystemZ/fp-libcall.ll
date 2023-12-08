; Test that library calls are emitted for LLVM IR intrinsics
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define float @f1(float %x, i32 %y) {
; CHECK-LABEL: f1:
; CHECK: brasl %r14, __powisf2@PLT
  %tmp = call float @llvm.powi.f32.i32(float %x, i32 %y)
  ret float %tmp
}

define double @f2(double %x, i32 %y) {
; CHECK-LABEL: f2:
; CHECK: brasl %r14, __powidf2@PLT
  %tmp = call double @llvm.powi.f64.i32(double %x, i32 %y)
  ret double %tmp
}

define fp128 @f3(fp128 %x, i32 %y) {
; CHECK-LABEL: f3:
; CHECK: brasl %r14, __powitf2@PLT
  %tmp = call fp128 @llvm.powi.f128.i32(fp128 %x, i32 %y)
  ret fp128 %tmp
}

define float @f4(float %x, float %y) {
; CHECK-LABEL: f4:
; CHECK: brasl %r14, powf@PLT
  %tmp = call float @llvm.pow.f32(float %x, float %y)
  ret float %tmp
}

define double @f5(double %x, double %y) {
; CHECK-LABEL: f5:
; CHECK: brasl %r14, pow@PLT
  %tmp = call double @llvm.pow.f64(double %x, double %y)
  ret double %tmp
}

define fp128 @f6(fp128 %x, fp128 %y) {
; CHECK-LABEL: f6:
; CHECK: brasl %r14, powl@PLT
  %tmp = call fp128 @llvm.pow.f128(fp128 %x, fp128 %y)
  ret fp128 %tmp
}

define float @f7(float %x) {
; CHECK-LABEL: f7:
; CHECK: brasl %r14, sinf@PLT
  %tmp = call float @llvm.sin.f32(float %x)
  ret float %tmp
}

define double @f8(double %x) {
; CHECK-LABEL: f8:
; CHECK: brasl %r14, sin@PLT
  %tmp = call double @llvm.sin.f64(double %x)
  ret double %tmp
}

define fp128 @f9(fp128 %x) {
; CHECK-LABEL: f9:
; CHECK: brasl %r14, sinl@PLT
  %tmp = call fp128 @llvm.sin.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f10(float %x) {
; CHECK-LABEL: f10:
; CHECK: brasl %r14, cosf@PLT
  %tmp = call float @llvm.cos.f32(float %x)
  ret float %tmp
}

define double @f11(double %x) {
; CHECK-LABEL: f11:
; CHECK: brasl %r14, cos@PLT
  %tmp = call double @llvm.cos.f64(double %x)
  ret double %tmp
}

define fp128 @f12(fp128 %x) {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, cosl@PLT
  %tmp = call fp128 @llvm.cos.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f13(float %x) {
; CHECK-LABEL: f13:
; CHECK: brasl %r14, expf@PLT
  %tmp = call float @llvm.exp.f32(float %x)
  ret float %tmp
}

define double @f14(double %x) {
; CHECK-LABEL: f14:
; CHECK: brasl %r14, exp@PLT
  %tmp = call double @llvm.exp.f64(double %x)
  ret double %tmp
}

define fp128 @f15(fp128 %x) {
; CHECK-LABEL: f15:
; CHECK: brasl %r14, expl@PLT
  %tmp = call fp128 @llvm.exp.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f16(float %x) {
; CHECK-LABEL: f16:
; CHECK: brasl %r14, exp2f@PLT
  %tmp = call float @llvm.exp2.f32(float %x)
  ret float %tmp
}

define double @f17(double %x) {
; CHECK-LABEL: f17:
; CHECK: brasl %r14, exp2@PLT
  %tmp = call double @llvm.exp2.f64(double %x)
  ret double %tmp
}

define fp128 @f18(fp128 %x) {
; CHECK-LABEL: f18:
; CHECK: brasl %r14, exp2l@PLT
  %tmp = call fp128 @llvm.exp2.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f19(float %x) {
; CHECK-LABEL: f19:
; CHECK: brasl %r14, logf@PLT
  %tmp = call float @llvm.log.f32(float %x)
  ret float %tmp
}

define double @f20(double %x) {
; CHECK-LABEL: f20:
; CHECK: brasl %r14, log@PLT
  %tmp = call double @llvm.log.f64(double %x)
  ret double %tmp
}

define fp128 @f21(fp128 %x) {
; CHECK-LABEL: f21:
; CHECK: brasl %r14, logl@PLT
  %tmp = call fp128 @llvm.log.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f22(float %x) {
; CHECK-LABEL: f22:
; CHECK: brasl %r14, log2f@PLT
  %tmp = call float @llvm.log2.f32(float %x)
  ret float %tmp
}

define double @f23(double %x) {
; CHECK-LABEL: f23:
; CHECK: brasl %r14, log2@PLT
  %tmp = call double @llvm.log2.f64(double %x)
  ret double %tmp
}

define fp128 @f24(fp128 %x) {
; CHECK-LABEL: f24:
; CHECK: brasl %r14, log2l@PLT
  %tmp = call fp128 @llvm.log2.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f25(float %x) {
; CHECK-LABEL: f25:
; CHECK: brasl %r14, log10f@PLT
  %tmp = call float @llvm.log10.f32(float %x)
  ret float %tmp
}

define double @f26(double %x) {
; CHECK-LABEL: f26:
; CHECK: brasl %r14, log10@PLT
  %tmp = call double @llvm.log10.f64(double %x)
  ret double %tmp
}

define fp128 @f27(fp128 %x) {
; CHECK-LABEL: f27:
; CHECK: brasl %r14, log10l@PLT
  %tmp = call fp128 @llvm.log10.f128(fp128 %x)
  ret fp128 %tmp
}

define float @f28(float %x, float %y) {
; CHECK-LABEL: f28:
; CHECK: brasl %r14, fminf@PLT
  %tmp = call float @llvm.minnum.f32(float %x, float %y)
  ret float %tmp
}

define double @f29(double %x, double %y) {
; CHECK-LABEL: f29:
; CHECK: brasl %r14, fmin@PLT
  %tmp = call double @llvm.minnum.f64(double %x, double %y)
  ret double %tmp
}

define fp128 @f30(fp128 %x, fp128 %y) {
; CHECK-LABEL: f30:
; CHECK: brasl %r14, fminl@PLT
  %tmp = call fp128 @llvm.minnum.f128(fp128 %x, fp128 %y)
  ret fp128 %tmp
}

define float @f31(float %x, float %y) {
; CHECK-LABEL: f31:
; CHECK: brasl %r14, fmaxf@PLT
  %tmp = call float @llvm.maxnum.f32(float %x, float %y)
  ret float %tmp
}

define double @f32(double %x, double %y) {
; CHECK-LABEL: f32:
; CHECK: brasl %r14, fmax@PLT
  %tmp = call double @llvm.maxnum.f64(double %x, double %y)
  ret double %tmp
}

define fp128 @f33(fp128 %x, fp128 %y) {
; CHECK-LABEL: f33:
; CHECK: brasl %r14, fmaxl@PLT
  %tmp = call fp128 @llvm.maxnum.f128(fp128 %x, fp128 %y)
  ret fp128 %tmp
}

; Verify that "nnan" minnum/maxnum calls are transformed to
; compare+select sequences instead of libcalls.
define float @f34(float %x, float %y) {
; CHECK-LABEL: f34:
; CHECK: cebr %f0, %f2
; CHECK: blr %r14
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %tmp = call nnan float @llvm.minnum.f32(float %x, float %y)
  ret float %tmp
}

define double @f35(double %x, double %y) {
; CHECK-LABEL: f35:
; CHECK: cdbr %f0, %f2
; CHECK: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %tmp = call nnan double @llvm.minnum.f64(double %x, double %y)
  ret double %tmp
}

define fp128 @f36(fp128 %x, fp128 %y) {
; CHECK-LABEL: f36:
; CHECK: cxbr
; CHECK: jl
; CHECK: lxr
; CHECK: br %r14
  %tmp = call nnan fp128 @llvm.minnum.f128(fp128 %x, fp128 %y)
  ret fp128 %tmp
}

define float @f37(float %x, float %y) {
; CHECK-LABEL: f37:
; CHECK: cebr %f0, %f2
; CHECK: bhr %r14
; CHECK: ler %f0, %f2
; CHECK: br %r14
  %tmp = call nnan float @llvm.maxnum.f32(float %x, float %y)
  ret float %tmp
}

define double @f38(double %x, double %y) {
; CHECK-LABEL: f38:
; CHECK: cdbr %f0, %f2
; CHECK: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %tmp = call nnan double @llvm.maxnum.f64(double %x, double %y)
  ret double %tmp
}

define fp128 @f39(fp128 %x, fp128 %y) {
; CHECK-LABEL: f39:
; CHECK: cxbr
; CHECK: jh
; CHECK: lxr
; CHECK: br %r14
  %tmp = call nnan fp128 @llvm.maxnum.f128(fp128 %x, fp128 %y)
  ret fp128 %tmp
}

declare float @llvm.powi.f32.i32(float, i32)
declare double @llvm.powi.f64.i32(double, i32)
declare fp128 @llvm.powi.f128.i32(fp128, i32)
declare float @llvm.pow.f32(float, float)
declare double @llvm.pow.f64(double, double)
declare fp128 @llvm.pow.f128(fp128, fp128)

declare float @llvm.sin.f32(float)
declare double @llvm.sin.f64(double)
declare fp128 @llvm.sin.f128(fp128)
declare float @llvm.cos.f32(float)
declare double @llvm.cos.f64(double)
declare fp128 @llvm.cos.f128(fp128)

declare float @llvm.exp.f32(float)
declare double @llvm.exp.f64(double)
declare fp128 @llvm.exp.f128(fp128)
declare float @llvm.exp2.f32(float)
declare double @llvm.exp2.f64(double)
declare fp128 @llvm.exp2.f128(fp128)

declare float @llvm.log.f32(float)
declare double @llvm.log.f64(double)
declare fp128 @llvm.log.f128(fp128)
declare float @llvm.log2.f32(float)
declare double @llvm.log2.f64(double)
declare fp128 @llvm.log2.f128(fp128)
declare float @llvm.log10.f32(float)
declare double @llvm.log10.f64(double)
declare fp128 @llvm.log10.f128(fp128)

declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare fp128 @llvm.minnum.f128(fp128, fp128)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare fp128 @llvm.maxnum.f128(fp128, fp128)

