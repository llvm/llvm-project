; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"
; CHECK: OpMemoryModel Logical GLSL450

define noundef half @test_fmax_half(half noundef %a, half noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] NMin %[[#]] %[[#]]
  %0 = call half @llvm.minnum.f16(half %a, half %b)
  ret half %0
}

define noundef float @test_fmax_float(float noundef %a, float noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] NMin %[[#]] %[[#]]
  %0 = call float @llvm.minnum.f32(float %a, float %b)
  ret float %0
}

define noundef double @test_fmax_double(double noundef %a, double noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] NMin %[[#]] %[[#]]
  %0 = call double @llvm.minnum.f64(double %a, double %b)
  ret double %0
}

declare half @llvm.minnum.f16(half, half)
declare float @llvm.minnum.f32(float, float)
declare double @llvm.minnum.f64(double, double)
