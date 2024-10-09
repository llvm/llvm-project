; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for saturate are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#vec4_float_64:]] = OpTypeVector %[[#float_64]] 4
; CHECK-DAG: %[[#zero_float_16:]] = OpConstant %[[#float_16]] 0
; CHECK-DAG: %[[#vec4_zero_float_16:]] = OpConstantComposite %[[#vec4_float_16]] %[[#zero_float_16]] %[[#zero_float_16]] %[[#zero_float_16]]
; CHECK-DAG: %[[#one_float_16:]] = OpConstant %[[#float_16]] 15360
; CHECK-DAG: %[[#vec4_one_float_16:]] = OpConstantComposite %[[#vec4_float_16]] %[[#one_float_16]] %[[#one_float_16]] %[[#one_float_16]]
; CHECK-DAG: %[[#zero_float_32:]] = OpConstant %[[#float_32]] 0
; CHECK-DAG: %[[#vec4_zero_float_32:]] = OpConstantComposite %[[#vec4_float_32]] %[[#zero_float_32]] %[[#zero_float_32]] %[[#zero_float_32]]
; CHECK-DAG: %[[#one_float_32:]] = OpConstant %[[#float_32]] 1
; CHECK-DAG: %[[#vec4_one_float_32:]] = OpConstantComposite %[[#vec4_float_32]] %[[#one_float_32]] %[[#one_float_32]] %[[#one_float_32]]

; CHECK-DAG: %[[#zero_float_64:]] = OpConstant %[[#float_64]] 0
; CHECK-DAG: %[[#vec4_zero_float_64:]] = OpConstantComposite %[[#vec4_float_64]] %[[#zero_float_64]] %[[#zero_float_64]] %[[#zero_float_64]]
; CHECK-DAG: %[[#one_float_64:]] = OpConstant %[[#float_64]] 1
; CHECK-DAG: %[[#vec4_one_float_64:]] = OpConstantComposite %[[#vec4_float_64]] %[[#one_float_64]] %[[#one_float_64]] %[[#one_float_64]]

define noundef half @saturate_half(half noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#zero_float_16]] %[[#one_float_16]]
  %hlsl.saturate = call half @llvm.spv.saturate.f16(half %a)
  ret half %hlsl.saturate
}

define noundef float @saturate_float(float noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#zero_float_32]] %[[#one_float_32]]
  %hlsl.saturate = call float @llvm.spv.saturate.f32(float %a)
  ret float %hlsl.saturate
}

define noundef double @saturate_double(double noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_64]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_64]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#zero_float_64]] %[[#one_float_64]]
  %hlsl.saturate = call double @llvm.spv.saturate.f64(double %a)
  ret double %hlsl.saturate
}

define noundef <4 x half> @saturate_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#vec4_zero_float_16]] %[[#vec4_one_float_16]]
  %hlsl.saturate = call <4 x half> @llvm.spv.saturate.v4f16(<4 x half> %a)
  ret <4 x half> %hlsl.saturate
}

define noundef <4 x float> @saturate_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#vec4_zero_float_32]] %[[#vec4_one_float_32]]
  %hlsl.saturate = call <4 x float> @llvm.spv.saturate.v4f32(<4 x float> %a)
  ret <4 x float> %hlsl.saturate
}

define noundef <4 x double> @saturate_double4(<4 x double> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_64]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_64]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_64]] %[[#op_ext_glsl]] FClamp %[[#arg0]] %[[#vec4_zero_float_64]] %[[#vec4_one_float_64]]
  %hlsl.saturate = call <4 x double> @llvm.spv.saturate.v4f64(<4 x double> %a)
  ret <4 x double> %hlsl.saturate
}

declare <4 x half> @llvm.spv.saturate.v4f16(<4 x half>)
declare <4 x float> @llvm.spv.saturate.v4f32(<4 x float>)
declare <4 x double> @llvm.spv.saturate.v4f64(<4 x double>)
