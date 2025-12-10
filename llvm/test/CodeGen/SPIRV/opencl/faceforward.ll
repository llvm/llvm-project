; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure we don't pattern match to faceforward in OpenCL.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "OpenCL.std"

; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#zero:]] = OpConstantNull %[[#float_32]]

define internal noundef float @faceforward_no_instcombine_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#fmul:]] = OpFMul %[[#float_32]] %[[#arg1]] %[[#arg2]]
  ; CHECK: %[[#fcmp:]] = OpFOrdLessThan %[[#bool]] %[[#fmul]] %[[#zero]]
  ; CHECK: %[[#fneg:]] = OpFNegate %[[#float_32]] %[[#arg0]]
  ; CHECK: %[[#select:]] = OpSelect %[[#float_32]] %[[#fcmp]] %[[#arg0]] %[[#fneg]]
  %fmul= fmul float %b, %c
  %fcmp = fcmp olt float %fmul, 0.000000e+00
  %fneg = fneg float %a
  %select = select i1 %fcmp, float %a, float %fneg
  ret float %select
 } 
