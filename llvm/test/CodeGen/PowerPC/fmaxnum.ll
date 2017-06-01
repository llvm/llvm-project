; RUN: llc -verify-machineinstrs -march=ppc32 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s

declare float @fmaxf(float, float)
declare double @fmax(double, double)
declare ppc_fp128 @fmaxl(ppc_fp128, ppc_fp128)
declare float @llvm.maxnum.f32(float, float)
declare double @llvm.maxnum.f64(double, double)
declare ppc_fp128 @llvm.maxnum.ppcf128(ppc_fp128, ppc_fp128)

declare <2 x float> @llvm.maxnum.v2f32(<2 x float>, <2 x float>)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare <8 x float> @llvm.maxnum.v8f32(<8 x float>, <8 x float>)

; CHECK-LABEL: @test_fmaxf
; CHECK: bl fmaxf
define float @test_fmaxf(float %x, float %y) {
  %z = call float @fmaxf(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_fmax
; CHECK: bl fmax
define double @test_fmax(double %x, double %y) {
  %z = call double @fmax(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_fmaxl
; CHECK: bl fmaxl
define ppc_fp128 @test_fmaxl(ppc_fp128 %x, ppc_fp128 %y) {
  %z = call ppc_fp128 @fmaxl(ppc_fp128 %x, ppc_fp128 %y) readnone
  ret ppc_fp128 %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf
; CHECK: bl fmaxf
define float @test_intrinsic_fmaxf(float %x, float %y) {
  %z = call float @llvm.maxnum.f32(float %x, float %y) readnone
  ret float %z
}

; CHECK-LABEL: @test_intrinsic_fmax
; CHECK: bl fmax
define double @test_intrinsic_fmax(double %x, double %y) {
  %z = call double @llvm.maxnum.f64(double %x, double %y) readnone
  ret double %z
}

; CHECK-LABEL: @test_intrinsic_fmaxl
; CHECK: bl fmaxl
define ppc_fp128 @test_intrinsic_fmaxl(ppc_fp128 %x, ppc_fp128 %y) {
  %z = call ppc_fp128 @llvm.maxnum.ppcf128(ppc_fp128 %x, ppc_fp128 %y) readnone
  ret ppc_fp128 %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf_v2f32
; CHECK: bl fmaxf
; CHECK: bl fmaxf
define <2 x float> @test_intrinsic_fmaxf_v2f32(<2 x float> %x, <2 x float> %y) {
  %z = call <2 x float> @llvm.maxnum.v2f32(<2 x float> %x, <2 x float> %y) readnone
  ret <2 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf_v4f32
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
define <4 x float> @test_intrinsic_fmaxf_v4f32(<4 x float> %x, <4 x float> %y) {
  %z = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %x, <4 x float> %y) readnone
  ret <4 x float> %z
}

; CHECK-LABEL: @test_intrinsic_fmaxf_v8f32
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
; CHECK: bl fmaxf
define <8 x float> @test_intrinsic_fmaxf_v8f32(<8 x float> %x, <8 x float> %y) {
  %z = call <8 x float> @llvm.maxnum.v8f32(<8 x float> %x, <8 x float> %y) readnone
  ret <8 x float> %z
}
