; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_cl:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef <4 x half> @normalize_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_cl]] normalize %[[#arg0]]
  %spv.normalize = call <4 x half> @llvm.spv.normalize.v4f16(<4 x half> %a)
  ret <4 x half> %spv.normalize
}

define noundef <4 x float> @normalize_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_cl]] normalize %[[#arg0]]
  %spv.normalize = call <4 x float> @llvm.spv.normalize.v4f32(<4 x float> %a)
  ret <4 x float> %spv.normalize
}

define noundef <4 x float> @normalize_instcombine_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_cl]] normalize %[[#arg0]]
  %spv.length = call float @llvm.spv.length.f32(<4 x float> %a)
  %splatinsert = insertelement <4 x float> poison, float %spv.length, i64 0
  %splat = shufflevector <4 x float> %splatinsert, <4 x float> poison, <4 x i32> zeroinitializer
  %div = fdiv <4 x float> %a, %splat
  ret <4 x float> %div
}

declare <4 x half> @llvm.spv.normalize.v4f16(<4 x half>)
declare <4 x float> @llvm.spv.normalize.v4f32(<4 x float>)
