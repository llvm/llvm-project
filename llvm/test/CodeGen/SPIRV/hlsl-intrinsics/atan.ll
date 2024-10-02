; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

define noundef float @atan_float(float noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Atan %[[#arg0]]
  %elt.atan = call float @llvm.atan.f32(float %a)
  ret float %elt.atan
}

define noundef half @atan_half(half noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Atan %[[#arg0]]
  %elt.atan = call half @llvm.atan.f16(half %a)
  ret half %elt.atan
}

define noundef <4 x float> @atan_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan %[[#arg0]]
  %elt.atan = call <4 x float> @llvm.atan.v4f32(<4 x float> %a)
  ret <4 x float> %elt.atan
}

define noundef <4 x half> @atan_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan %[[#arg0]]
  %elt.atan = call <4 x half> @llvm.atan.v4f16(<4 x half> %a)
  ret <4 x half> %elt.atan
}

declare half @llvm.atan.f16(half)
declare float @llvm.atan.f32(float)
declare <4 x half> @llvm.atan.v4f16(<4 x half>)
declare <4 x float> @llvm.atan.v4f32(<4 x float>)
