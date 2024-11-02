; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for length are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef half @length_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Length %[[#arg0]]
  %hlsl.length = call half @llvm.spv.length.v4f16(<4 x half> %a)
  ret half %hlsl.length
}

define noundef float @length_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Length %[[#arg0]]
  %hlsl.length = call float @llvm.spv.length.v4f32(<4 x float> %a)
  ret float %hlsl.length
}

declare half @llvm.spv.length.v4f16(<4 x half>)
declare float @llvm.spv.length.v4f32(<4 x float>)
