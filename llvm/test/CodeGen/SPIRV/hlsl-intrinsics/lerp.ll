; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for lerp are generated as FMix

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef half @lerp_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] FMix %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %hlsl.lerp = call half @llvm.spv.lerp.f16(half %a, half %b, half %c)
  ret half %hlsl.lerp
}


define noundef float @lerp_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FMix %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %hlsl.lerp = call float @llvm.spv.lerp.f32(float %a, float %b, float %c)
  ret float %hlsl.lerp
}

define noundef <4 x half> @lerp_half4(<4 x half> noundef %a, <4 x half> noundef %b, <4 x half> noundef %c) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] FMix %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %hlsl.lerp = call <4 x half> @llvm.spv.lerp.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %hlsl.lerp
}

define noundef <4 x float> @lerp_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FMix %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %hlsl.lerp = call <4 x float> @llvm.spv.lerp.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %hlsl.lerp
}

declare half @llvm.spv.lerp.f16(half, half, half)
declare float @llvm.spv.lerp.f32(float, float, float)
declare <4 x half> @llvm.spv.lerp.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.lerp.v4f32(<4 x float>, <4 x float>, <4 x float>)
