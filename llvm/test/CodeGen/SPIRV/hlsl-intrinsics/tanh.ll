; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

define noundef float @tanh_float(float noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Tanh %[[#arg0]]
  %elt.tanh = call float @llvm.tanh.f32(float %a)
  ret float %elt.tanh
}

define noundef half @tanh_half(half noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Tanh %[[#arg0]]
  %elt.tanh = call half @llvm.tanh.f16(half %a)
  ret half %elt.tanh
}

define noundef <4 x float> @tanh_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Tanh %[[#arg0]]
  %elt.tanh = call <4 x float> @llvm.tanh.v4f32(<4 x float> %a)
  ret <4 x float> %elt.tanh
}

define noundef <4 x half> @tanh_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Tanh %[[#arg0]]
  %elt.tanh = call <4 x half> @llvm.tanh.v4f16(<4 x half> %a)
  ret <4 x half> %elt.tanh
}

declare half @llvm.tanh.f16(half)
declare float @llvm.tanh.f32(float)
declare <4 x half> @llvm.tanh.v4f16(<4 x half>)
declare <4 x float> @llvm.tanh.v4f32(<4 x float>)
