; RUN: llc -verify-machineinstrs -march=ppc32 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

declare float @fminf(float, float)
declare double @fmin(double, double)
declare ppc_fp128 @fminl(ppc_fp128, ppc_fp128)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
declare ppc_fp128 @llvm.minnum.ppcf128(ppc_fp128, ppc_fp128)

declare <2 x float> @llvm.minnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.minnum.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.minnum.v8f32(<8 x float>, <8 x float>)

; CHECK-LABEL: @test_fminf
; CHECK: bl fminf
define float @test_fminf(float %x, float %y) {
  %z = call float @fminf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmin
; CHECK: bl fmin
define double @test_fmin(double %x, double %y) {
  %z = call double @fmin(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fminl
; CHECK: bl fminl
define ppc_fp128 @test_fminl(ppc_fp128 %x, ppc_fp128 %y) {
  %z = call ppc_fp128 @fminl(ppc_fp128 %x, ppc_fp128 %y) readnone
  ret ppc_fp128 %z
}

; CHECK-LABEL: @test_intrinsic_fmin_f32
; CHECK: bl fminf
define float @test_intrinsic_fmin_f32(float %x, float %y) {
  %z = call float @llvm.minnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmin_f64
; CHECK: bl fmin
define double @test_intrinsic_fmin_f64(double %x, double %y) {
  %z = call double @llvm.minnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fmin_f128
; CHECK: bl fminl
define ppc_fp128 @test_intrinsic_fmin_f128(ppc_fp128 %x, ppc_fp128 %y) {
  %z = call ppc_fp128 @llvm.minnum.ppcf128(ppc_fp128 %x, ppc_fp128 %y) readnone
  ret ppc_fp128 %z
}

; CHECK-LABEL: @test_intrinsic_fminf_v2f32
; CHECK: bl fminf
; CHECK: bl fminf
define <2 x float> @test_intrinsic_fminf_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.minnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v4f32
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
define <4 x float> @test_intrinsic_fmin_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.minnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmin_v8f32
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
; CHECK: bl fminf
define <8 x float> @test_intrinsic_fmin_v8f32(<8 x float> %x, <8 x float> %y) {
  %z = call <8 x float> @llvm.minnum.v8f32(<8 x float> %x, <8 x float> %y) readnone
  ret <8 x float> %z
}
