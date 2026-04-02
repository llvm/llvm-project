; RUN: llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for refract are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef  <4 x half> @refract_half(<4 x half> noundef  %I, <4 x half> noundef  %N, half noundef  %ETA) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_16:]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Refract %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.refract.i = tail call reassoc nnan ninf nsz arcp afn noundef <4 x half> @llvm.spv.refract.v4f16.f16(<4 x half> %I, <4 x half> %N, half %ETA)
  ret <4 x half> %spv.refract.i
}

define noundef  <4 x float> @refract_float4(<4 x float> noundef  %I, <4 x float> noundef  %N, float noundef  %ETA) {
entry:
  %conv.i = fpext reassoc nnan ninf nsz arcp afn float %ETA to double
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_32:]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Refract %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.refract.i = tail call reassoc nnan ninf nsz arcp afn noundef <4 x float> @llvm.spv.refract.v4f32.f32(<4 x float> %I, <4 x float> %N, float %ETA)
  ret <4 x float> %spv.refract.i
}

declare <4 x half> @llvm.spv.refract.v4f16.f16(<4 x half>, <4 x half>, half)
declare <4 x float> @llvm.spv.reflect.v4f32.f32(<4 x float>, <4 x float>, float)
