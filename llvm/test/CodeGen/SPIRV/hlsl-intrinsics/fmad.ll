; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef half @fmad_half(half noundef %a, half noundef %b, half noundef %c) #0 {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %hlsl.fmad = call half @llvm.fmuladd.f16(half %a, half %b, half %c)
  ret half %hlsl.fmad
}

define noundef float @fmad_float(float noundef %a, float noundef %b, float noundef %c) #0 {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %hlsl.fmad = call float @llvm.fmuladd.f32(float %a, float %b, float %c)
  ret float %hlsl.fmad
}

define noundef double @fmad_double(double noundef %a, double noundef %b, double noundef %c) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Fma %[[#]] %[[#]] %[[#]]
  %hlsl.fmad = call double @llvm.fmuladd.f64(double %a, double %b, double %c)
  ret double %hlsl.fmad
}

declare half @llvm.fmuladd.f16(half, half, half)
declare float @llvm.fmuladd.f32(float, float, float)
declare double @llvm.fmuladd.f64(double, double, double)
