; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; Vulkan SPIR-V has no copysign in GLSL.std.450, so llvm.copysign is lowered with bit manipulation.

; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#int_16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#sign_16:]] = OpConstant %[[#int_16]] 32768
; CHECK-DAG: %[[#magn_16:]] = OpConstant %[[#int_16]] 32767
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#int_32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#sign_32:]] = OpConstant %[[#int_32]] 2147483648
; CHECK-DAG: %[[#magn_32:]] = OpConstant %[[#int_32]] 2147483647
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_int_32:]] = OpTypeVector %[[#int_32]] 4
; CHECK-DAG: %[[#vec4_sign_32:]] = OpConstantComposite %[[#vec4_int_32]] %[[#sign_32]] %[[#sign_32]] %[[#sign_32]] %[[#sign_32]]
; CHECK-DAG: %[[#vec4_magn_32:]] = OpConstantComposite %[[#vec4_int_32]] %[[#magn_32]] %[[#magn_32]] %[[#magn_32]] %[[#magn_32]]
; CHECK-DAG: %[[#float_64:]] = OpTypeFloat 64
; CHECK-DAG: %[[#int_64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#sign_64:]] = OpConstant %[[#int_64]] 9223372036854775808
; CHECK-DAG: %[[#magn_64:]] = OpConstant %[[#int_64]] 9223372036854775807

define noundef half @copysign_half(half noundef %a, half noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#arg0_int:]] = OpBitcast %[[#int_16]] %[[#arg0]]
  ; CHECK: %[[#arg1_int:]] = OpBitcast %[[#int_16]] %[[#arg1]]
  ; CHECK: %[[#arg0_magn:]] = OpBitwiseAnd %[[#int_16]] %[[#arg0_int]] %[[#magn_16]]
  ; CHECK: %[[#arg1_sign:]] = OpBitwiseAnd %[[#int_16]] %[[#arg1_int]] %[[#sign_16]]
  ; CHECK: %[[#combined:]] = OpBitwiseOr %[[#int_16]] %[[#arg0_magn]] %[[#arg1_sign]]
  ; CHECK: %[[#]] = OpBitcast %[[#float_16]] %[[#combined]]
  %r = call half @llvm.copysign.f16(half %a, half %b)
  ret half %r
}

define noundef float @copysign_float(float noundef %a, float noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg0_int:]] = OpBitcast %[[#int_32]] %[[#arg0]]
  ; CHECK: %[[#arg1_int:]] = OpBitcast %[[#int_32]] %[[#arg1]]
  ; CHECK: %[[#arg0_magn:]] = OpBitwiseAnd %[[#int_32]] %[[#arg0_int]] %[[#magn_32]]
  ; CHECK: %[[#arg1_sign:]] = OpBitwiseAnd %[[#int_32]] %[[#arg1_int]] %[[#sign_32]]
  ; CHECK: %[[#combined:]] = OpBitwiseOr %[[#int_32]] %[[#arg0_magn]] %[[#arg1_sign]]
  ; CHECK: %[[#]] = OpBitcast %[[#float_32]] %[[#combined]]
  %r = call float @llvm.copysign.f32(float %a, float %b)
  ret float %r
}

define noundef double @copysign_double(double noundef %a, double noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_64]]
  ; CHECK: %[[#arg0_int:]] = OpBitcast %[[#int_64]] %[[#arg0]]
  ; CHECK: %[[#arg1_int:]] = OpBitcast %[[#int_64]] %[[#arg1]]
  ; CHECK: %[[#arg0_magn:]] = OpBitwiseAnd %[[#int_64]] %[[#arg0_int]] %[[#magn_64]]
  ; CHECK: %[[#arg1_sign:]] = OpBitwiseAnd %[[#int_64]] %[[#arg1_int]] %[[#sign_64]]
  ; CHECK: %[[#combined:]] = OpBitwiseOr %[[#int_64]] %[[#arg0_magn]] %[[#arg1_sign]]
  ; CHECK: %[[#]] = OpBitcast %[[#float_64]] %[[#combined]]
  %r = call double @llvm.copysign.f64(double %a, double %b)
  ret double %r
}

define noundef <4 x float> @copysign_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg0_int:]] = OpBitcast %[[#vec4_int_32]] %[[#arg0]]
  ; CHECK: %[[#arg1_int:]] = OpBitcast %[[#vec4_int_32]] %[[#arg1]]
  ; CHECK: %[[#arg0_magn:]] = OpBitwiseAnd %[[#vec4_int_32]] %[[#arg0_int]] %[[#vec4_magn_32]]
  ; CHECK: %[[#arg1_sign:]] = OpBitwiseAnd %[[#vec4_int_32]] %[[#arg1_int]] %[[#vec4_sign_32]]
  ; CHECK: %[[#combined:]] = OpBitwiseOr %[[#vec4_int_32]] %[[#arg0_magn]] %[[#arg1_sign]]
  ; CHECK: %[[#]] = OpBitcast %[[#vec4_float_32]] %[[#combined]]
  %r = call <4 x float> @llvm.copysign.v4f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %r
}

declare half @llvm.copysign.f16(half, half)
declare float @llvm.copysign.f32(float, float)
declare double @llvm.copysign.f64(double, double)
declare <4 x float> @llvm.copysign.v4f32(<4 x float>, <4 x float>)
