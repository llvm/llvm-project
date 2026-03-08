; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef half @fma_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call half @llvm.spv.fma.f16(half %a, half %b, half %c)
  ret half %r
}

define noundef float @fma_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call float @llvm.spv.fma.f32(float %a, float %b, float %c)
  ret float %r
}

define noundef double @fma_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call double @llvm.spv.fma.f64(double %a, double %b, double %c)
  ret double %r
}

define noundef <4 x half> @fma_half4(<4 x half> noundef %a, <4 x half> noundef %b, <4 x half> noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call <4 x half> @llvm.spv.fma.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %r
}

define noundef <4 x float> @fma_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call <4 x float> @llvm.spv.fma.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %r
}

define noundef <4 x double> @fma_double4(<4 x double> noundef %a, <4 x double> noundef %b, <4 x double> noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %r = call <4 x double> @llvm.spv.fma.v4f64(<4 x double> %a, <4 x double> %b, <4 x double> %c)
  ret <4 x double> %r
}

declare half @llvm.spv.fma.f16(half, half, half)
declare float @llvm.spv.fma.f32(float, float, float)
declare double @llvm.spv.fma.f64(double, double, double)
declare <4 x half> @llvm.spv.fma.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.fma.v4f32(<4 x float>, <4 x float>, <4 x float>)
declare <4 x double> @llvm.spv.fma.v4f64(<4 x double>, <4 x double>, <4 x double>)
