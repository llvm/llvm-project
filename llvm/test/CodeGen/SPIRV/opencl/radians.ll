; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#op_ext_ocl:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16

; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4

declare half @llvm.spv.radians.f16(half)
declare float @llvm.spv.radians.f32(float)

declare <4 x float> @llvm.spv.radians.v4f32(<4 x float>)
declare <4 x half> @llvm.spv.radians.v4f16(<4 x half>)

define noundef float @radians_float(float noundef %a) {
entry:
; CHECK: %[[#float_32_arg:]] = OpFunctionParameter %[[#float_32]]
; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_ocl]] radians %[[#float_32_arg]]
  %elt.radians = call float @llvm.spv.radians.f32(float %a)
  ret float %elt.radians
}

define noundef half @radians_half(half noundef %a) {
entry:
; CHECK: %[[#float_16_arg:]] = OpFunctionParameter %[[#float_16]]
; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_ocl]] radians %[[#float_16_arg]]
  %elt.radians = call half @llvm.spv.radians.f16(half %a)
  ret half %elt.radians
}

define noundef <4 x float> @radians_float_vector(<4 x float> noundef %a) {
entry:
; CHECK: %[[#vec4_float_32_arg:]] = OpFunctionParameter %[[#vec4_float_32]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_ocl]] radians %[[#vec4_float_32_arg]]
  %elt.radians = call <4 x float> @llvm.spv.radians.v4f32(<4 x float> %a)
  ret <4 x float> %elt.radians
}

define noundef <4 x half> @radians_half_vector(<4 x half> noundef %a) {
entry:
; CHECK: %[[#vec4_float_16_arg:]] = OpFunctionParameter %[[#vec4_float_16]]
; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_ocl]] radians %[[#vec4_float_16_arg]]
  %elt.radians = call <4 x half> @llvm.spv.radians.v4f16(<4 x half> %a)
  ret <4 x half> %elt.radians
}

