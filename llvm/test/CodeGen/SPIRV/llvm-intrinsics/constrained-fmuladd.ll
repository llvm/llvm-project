; RUN: llc -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTZ
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTP
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTN
; CHECK-DAG: OpDecorate %[[#]] FPRoundingMode RTE

; CHECK: OpFMul %[[#]] %[[#]] %[[#]]
; CHECK: OpFAdd %[[#]] %[[#]] %[[#]]
define spir_kernel void @test_f32(float %a) {
entry:
  %r = tail call float @llvm.experimental.constrained.fmuladd.f32(
              float %a, float %a, float %a,
              metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret void
}

; CHECK: OpFMul %[[#]] %[[#]] %[[#]] 
; CHECK: OpFAdd %[[#]] %[[#]] %[[#]]
define spir_kernel void @test_f64(double %a) {
entry:
  %r = tail call double @llvm.experimental.constrained.fmuladd.f64(
              double %a, double %a, double %a,
              metadata !"round.towardzero", metadata !"fpexcept.strict")
  ret void
}

; CHECK: OpFMul %[[#]] %[[#]] %[[#]]
; CHECK: OpFAdd %[[#]] %[[#]] %[[#]]
define spir_kernel void @test_v2f32(<2 x float> %a) {
entry:
  %r = tail call <2 x float> @llvm.experimental.constrained.fmuladd.v2f32(
              <2 x float> %a, <2 x float> %a, <2 x float> %a,
              metadata !"round.upward", metadata !"fpexcept.strict")
  ret void
}

; CHECK: OpFMul %[[#]] %[[#]] %[[#]]
; CHECK: OpFAdd %[[#]] %[[#]] %[[#]]
define spir_kernel void @test_v4f32(<4 x float> %a) {
entry:
  %r = tail call <4 x float> @llvm.experimental.constrained.fmuladd.v4f32(
              <4 x float> %a, <4 x float> %a, <4 x float> %a,
              metadata !"round.downward", metadata !"fpexcept.strict")
  ret void
}

; CHECK: OpFMul %[[#]] %[[#]] %[[#]]
; CHECK: OpFAdd %[[#]] %[[#]] %[[#]]
define spir_kernel void @test_v2f64(<2 x double> %a) {
entry:
  %r = tail call <2 x double> @llvm.experimental.constrained.fmuladd.v2f64(
              <2 x double> %a, <2 x double> %a, <2 x double> %a,
              metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret void
}

declare float  @llvm.experimental.constrained.fmuladd.f32(float, float, float, metadata, metadata)
declare double @llvm.experimental.constrained.fmuladd.f64(double, double, double, metadata, metadata)
declare <2 x float> @llvm.experimental.constrained.fmuladd.v2f32(<2 x float>, <2 x float>, <2 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fmuladd.v4f32(<4 x float>, <4 x float>, <4 x float>, metadata, metadata)
declare <2 x double> @llvm.experimental.constrained.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>, metadata, metadata)
