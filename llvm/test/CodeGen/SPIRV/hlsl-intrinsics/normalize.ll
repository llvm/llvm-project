; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for normalize are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef <4 x half> @normalize_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Normalize %[[#arg0]]
  %hlsl.normalize = call <4 x half> @llvm.spv.normalize.v4f16(<4 x half> %a)
  ret <4 x half> %hlsl.normalize
}

define noundef <4 x float> @normalize_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Normalize %[[#arg0]]
  %hlsl.normalize = call <4 x float> @llvm.spv.normalize.v4f32(<4 x float> %a)
  ret <4 x float> %hlsl.normalize
}

declare <4 x half> @llvm.spv.normalize.v4f16(<4 x half>)
declare <4 x float> @llvm.spv.normalize.v4f32(<4 x float>)
