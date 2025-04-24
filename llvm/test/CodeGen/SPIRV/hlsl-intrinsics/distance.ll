; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for distance are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef half @distance_half4(<4 x half> noundef %a, <4 x half> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Distance %[[#arg0]] %[[#arg1]]
  %spv.distance = call half @llvm.spv.distance.f16(<4 x half> %a, <4 x half> %b)
  ret half %spv.distance
}

define noundef float @distance_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Distance %[[#arg0]] %[[#arg1]]
  %spv.distance = call float @llvm.spv.distance.f32(<4 x float> %a, <4 x float> %b)
  ret float %spv.distance
}

define noundef float @distance_instcombine_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Distance %[[#arg0]] %[[#arg1]]
  %delta = fsub  <4 x float> %a, %b
  %spv.length = call float @llvm.spv.length.f32(<4 x float> %delta)
  ret float %spv.length
}

declare half @llvm.spv.distance.f16(<4 x half>, <4 x half>)
declare float @llvm.spv.distance.f32(<4 x float>, <4 x float>)
