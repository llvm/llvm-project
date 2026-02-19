; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16

; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

define noundef float @ddx_fine_float(float noundef %a) {
entry:
; CHECK: %[[#float_32_arg:]] = OpFunctionParameter %[[#float_32]]
; CHECK: %[[#]] = OpDPdxFine %[[#float_32]] %[[#float_32_arg]]
  %elt.ddx.fine = call float @llvm.spv.ddx.fine.f32(float %a)
  ret float %elt.ddx.fine
}

define noundef half @ddx_fine_half(half noundef %a) {
entry:
; CHECK: %[[#float_16_arg:]] = OpFunctionParameter %[[#float_16]]
; CHECK: %[[#converted:]] = OpFConvert %[[#float_32:]] %[[#float_16_arg]]
; CHECK: %[[#fine:]] = OpDPdxFine %[[#float_32]] %[[#converted]]
; CHECK: %[[#]] = OpFConvert %[[#float_16]] %[[#fine]]
  %elt.ddx.fine = call half @llvm.spv.ddx.fine.f16(half %a)
  ret half %elt.ddx.fine
}

define noundef <4 x float> @ddx_fine_float_vector(<4 x float> noundef %a) {
entry:
; CHECK: %[[#vec4_float_32_arg:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: %[[#]] = OpDPdxFine %[[#vec4_float_32]] %[[#vec4_float_32_arg]]
  %elt.ddx.fine = call <4 x float> @llvm.spv.ddx.fine.v4f32(<4 x float> %a)
  ret <4 x float> %elt.ddx.fine
}

define noundef <4 x half> @ddx_fine_half_vector(<4 x half> noundef %a) {
entry:
; CHECK: %[[#vec4_float_16_arg:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: %[[#converted:]] = OpFConvert %[[#vec4_float_32:]] %[[#vec4_float_16_arg]]
; CHECK: %[[#fine:]] = OpDPdxFine %[[#vec4_float_32]] %[[#converted]]
; CHECK: %[[#]] = OpFConvert %[[#vec4_float_16]] %[[#fine]]
  %elt.ddx.fine = call <4 x half> @llvm.spv.ddx.fine.v4f16(<4 x half> %a)
  ret <4 x half> %elt.ddx.fine
}

declare float @llvm.spv.ddx.fine.f32(float)
declare half @llvm.spv.ddx.fine.f16(half)
