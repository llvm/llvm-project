; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure dxil operation function calls for dot are generated for float type vectors.

; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec2_float_16:]] = OpTypeVector %[[#float_16]] 2
; CHECK-DAG: %[[#vec3_float_16:]] = OpTypeVector %[[#float_16]] 3
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec2_float_32:]] = OpTypeVector %[[#float_32]] 2
; CHECK-DAG: %[[#vec3_float_32:]] = OpTypeVector %[[#float_32]] 3
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4


define noundef half @dot_half2(<2 x half> noundef %a, <2 x half> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_float_16]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec2_float_16]]
; CHECK: OpDot %[[#float_16]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call half @llvm.spv.fdot.v2f16(<2 x half> %a, <2 x half> %b)
  ret half %dx.dot
}

define noundef half @dot_half3(<3 x half> noundef %a, <3 x half> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_16]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_16]]
; CHECK: OpDot %[[#float_16]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call half @llvm.spv.fdot.v3f16(<3 x half> %a, <3 x half> %b)
  ret half %dx.dot
}

define noundef half @dot_half4(<4 x half> noundef %a, <4 x half> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: OpDot %[[#float_16]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call half @llvm.spv.fdot.v4f16(<4 x half> %a, <4 x half> %b)
  ret half %dx.dot
}

define noundef float @dot_float2(<2 x float> noundef %a, <2 x float> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec2_float_32]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec2_float_32]]
; CHECK: OpDot %[[#float_32]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call float @llvm.spv.fdot.v2f32(<2 x float> %a, <2 x float> %b)
  ret float %dx.dot
}

define noundef float @dot_float3(<3 x float> noundef %a, <3 x float> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_32]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_32]]
; CHECK: OpDot %[[#float_32]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call float @llvm.spv.fdot.v3f32(<3 x float> %a, <3 x float> %b)
  ret float %dx.dot
}

define noundef float @dot_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: OpDot %[[#float_32]] %[[#arg0:]] %[[#arg1:]]
  %dx.dot = call float @llvm.spv.fdot.v4f32(<4 x float> %a, <4 x float> %b)
  ret float %dx.dot
}

declare half  @llvm.spv.fdot.v2f16(<2 x half> , <2 x half> )
declare half  @llvm.spv.fdot.v3f16(<3 x half> , <3 x half> )
declare half  @llvm.spv.fdot.v4f16(<4 x half> , <4 x half> )
declare float @llvm.spv.fdot.v2f32(<2 x float>, <2 x float>)
declare float @llvm.spv.fdot.v3f32(<3 x float>, <3 x float>)
declare float @llvm.spv.fdot.v4f32(<4 x float>, <4 x float>)
