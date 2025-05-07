; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

define noundef float @asin_float(float noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] Asin %[[#arg0]]
  %elt.asin = call float @llvm.asin.f32(float %a)
  ret float %elt.asin
}

define noundef half @asin_half(half noundef %a) {
entry:
; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] Asin %[[#arg0]]
  %elt.asin = call half @llvm.asin.f16(half %a)
  ret half %elt.asin
}

define noundef <4 x float> @asin_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] Asin %[[#arg0]]
  %elt.asin = call <4 x float> @llvm.asin.v4f32(<4 x float> %a)
  ret <4 x float> %elt.asin
}

define noundef <4 x half> @asin_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] Asin %[[#arg0]]
  %elt.asin = call <4 x half> @llvm.asin.v4f16(<4 x half> %a)
  ret <4 x half> %elt.asin
}

declare half @llvm.asin.f16(half)
declare float @llvm.asin.f32(float)
declare <4 x half> @llvm.asin.v4f16(<4 x half>)
declare <4 x float> @llvm.asin.v4f32(<4 x float>)
