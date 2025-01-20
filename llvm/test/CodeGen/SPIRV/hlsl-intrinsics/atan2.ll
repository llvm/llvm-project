; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

define noundef float @atan2_float(float noundef %a, float noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Atan2 %[[#arg0]] %[[#arg1]]
  %elt.atan2 = call float @llvm.atan2.f32(float %a, float %b)
  ret float %elt.atan2
}

define noundef half @atan2_half(half noundef %a, half noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Atan2 %[[#arg0]] %[[#arg1]]
  %elt.atan2 = call half @llvm.atan2.f16(half %a, half %b)
  ret half %elt.atan2
}

define noundef <4 x float> @atan2_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Atan2 %[[#arg0]] %[[#arg1]]
  %elt.atan2 = call <4 x float> @llvm.atan2.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %elt.atan2
}

define noundef <4 x half> @atan2_half4(<4 x half> noundef %a, <4 x half> noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Atan2 %[[#arg0]] %[[#arg1]]
  %elt.atan2 = call <4 x half> @llvm.atan2.v4f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %elt.atan2
}

declare half @llvm.atan2.f16(half, half)
declare float @llvm.atan2.f32(float, float)
declare <4 x half> @llvm.atan2.v4f16(<4 x half>, <4 x half>)
declare <4 x float> @llvm.atan2.v4f32(<4 x float>, <4 x float>)
