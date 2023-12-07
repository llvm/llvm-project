; RUN: llc < %s -asm-verbose=false -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define float @test_f32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_f32:
;CHECK: fmadd
;CHECK-NOT: fmadd
  %tmp1 = load float, ptr %A
  %tmp2 = load float, ptr %B
  %tmp3 = load float, ptr %C
  %tmp4 = call float @llvm.fmuladd.f32(float %tmp1, float %tmp2, float %tmp3)
  ret float %tmp4
}

define <2 x float> @test_v2f32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_v2f32:
;CHECK: fmla.2s
;CHECK-NOT: fmla.2s
  %tmp1 = load <2 x float>, ptr %A
  %tmp2 = load <2 x float>, ptr %B
  %tmp3 = load <2 x float>, ptr %C
  %tmp4 = call <2 x float> @llvm.fmuladd.v2f32(<2 x float> %tmp1, <2 x float> %tmp2, <2 x float> %tmp3)
  ret <2 x float> %tmp4
}

define <4 x float> @test_v4f32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_v4f32:
;CHECK: fmla.4s
;CHECK-NOT: fmla.4s
  %tmp1 = load <4 x float>, ptr %A
  %tmp2 = load <4 x float>, ptr %B
  %tmp3 = load <4 x float>, ptr %C
  %tmp4 = call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %tmp1, <4 x float> %tmp2, <4 x float> %tmp3)
  ret <4 x float> %tmp4
}

define <8 x float> @test_v8f32(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_v8f32:
;CHECK: fmla.4s
;CHECK: fmla.4s
;CHECK-NOT: fmla.4s
  %tmp1 = load <8 x float>, ptr %A
  %tmp2 = load <8 x float>, ptr %B
  %tmp3 = load <8 x float>, ptr %C
  %tmp4 = call <8 x float> @llvm.fmuladd.v8f32(<8 x float> %tmp1, <8 x float> %tmp2, <8 x float> %tmp3)
  ret <8 x float> %tmp4
}

define double @test_f64(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_f64:
;CHECK: fmadd
;CHECK-NOT: fmadd
  %tmp1 = load double, ptr %A
  %tmp2 = load double, ptr %B
  %tmp3 = load double, ptr %C
  %tmp4 = call double @llvm.fmuladd.f64(double %tmp1, double %tmp2, double %tmp3)
  ret double %tmp4
}

define <2 x double> @test_v2f64(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_v2f64:
;CHECK: fmla.2d
;CHECK-NOT: fmla.2d
  %tmp1 = load <2 x double>, ptr %A
  %tmp2 = load <2 x double>, ptr %B
  %tmp3 = load <2 x double>, ptr %C
  %tmp4 = call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %tmp1, <2 x double> %tmp2, <2 x double> %tmp3)
  ret <2 x double> %tmp4
}

define <4 x double> @test_v4f64(ptr %A, ptr %B, ptr %C) nounwind {
;CHECK-LABEL: test_v4f64:
;CHECK: fmla.2d
;CHECK: fmla.2d
;CHECK-NOT: fmla.2d
  %tmp1 = load <4 x double>, ptr %A
  %tmp2 = load <4 x double>, ptr %B
  %tmp3 = load <4 x double>, ptr %C
  %tmp4 = call <4 x double> @llvm.fmuladd.v4f64(<4 x double> %tmp1, <4 x double> %tmp2, <4 x double> %tmp3)
  ret <4 x double> %tmp4
}

declare float @llvm.fmuladd.f32(float, float, float) nounwind readnone
declare <2 x float> @llvm.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
declare <8 x float> @llvm.fmuladd.v8f32(<8 x float>, <8 x float>, <8 x float>) nounwind readnone
declare double @llvm.fmuladd.f64(double, double, double) nounwind readnone
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) nounwind readnone
declare <4 x double> @llvm.fmuladd.v4f64(<4 x double>, <4 x double>, <4 x double>) nounwind readnone
