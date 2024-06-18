; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"

; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64

; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4

define noundef float @rsqrt_float(float noundef %a) {
entry:
; CHECK: %[[#float_32_arg:]] = OpFunctionParameter %[[#float_32]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] InverseSqrt %[[#float_32_arg]]
  %elt.rsqrt = call float @llvm.spv.rsqrt.f32(float %a)
  ret float %elt.rsqrt
}

define noundef half @rsqrt_half(half noundef %a) {
entry:
; CHECK: %[[#float_16_arg:]] = OpFunctionParameter %[[#float_16]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] InverseSqrt %[[#float_16_arg]]
  %elt.rsqrt = call half @llvm.spv.rsqrt.f16(half %a)
  ret half %elt.rsqrt
}

define noundef double @rsqrt_double(double noundef %a) {
entry:
; CHECK: %[[#float_64_arg:]] = OpFunctionParameter %[[#float_64]]
; CHECK: %[[#]] = OpExtInst %[[#float_64]] %[[#op_ext_glsl]] InverseSqrt %[[#float_64_arg]]
  %elt.rsqrt = call double @llvm.spv.rsqrt.f64(double %a)
  ret double %elt.rsqrt
}

define noundef <4 x float> @rsqrt_float_vector(<4 x float> noundef %a) {
entry:
; CHECK: %[[#vec4_float_32_arg:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] InverseSqrt %[[#vec4_float_32_arg]]
  %elt.rsqrt = call <4 x float> @llvm.spv.rsqrt.v4f32(<4 x float> %a)
  ret <4 x float> %elt.rsqrt
}

define noundef <4 x half> @rsqrt_half_vector(<4 x half> noundef %a) {
entry:
; CHECK: %[[#vec4_float_16_arg:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] InverseSqrt %[[#vec4_float_16_arg]]
  %elt.rsqrt = call <4 x half> @llvm.spv.rsqrt.v4f16(<4 x half> %a)
  ret <4 x half> %elt.rsqrt
}

define noundef <4 x double> @rsqrt_double_vector(<4 x double> noundef %a) {
entry:
; CHECK: %[[#vec4_float_64_arg:]] = OpFunctionParameter %[[#vec4_float_64]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_float_64]] %[[#op_ext_glsl]] InverseSqrt %[[#vec4_float_64_arg]]
  %elt.rsqrt = call <4 x double> @llvm.spv.rsqrt.v4f64(<4 x double> %a)
  ret <4 x double> %elt.rsqrt
}

declare half @llvm.spv.rsqrt.f16(half)
declare float @llvm.spv.rsqrt.f32(float)
declare double @llvm.spv.rsqrt.f64(double)

declare <4 x float> @llvm.spv.rsqrt.v4f32(<4 x float>)
declare <4 x half> @llvm.spv.rsqrt.v4f16(<4 x half>)
declare <4 x double> @llvm.spv.rsqrt.v4f64(<4 x double>)
