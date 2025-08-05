; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.6 %}
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.6 %}

; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG:  %[[#vec_4_bool:]] = OpTypeVector %[[#bool]] 4

define spir_func float @opselect_float_scalar_test(float %x, float %y) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#fcmp:]] = OpFOrdGreaterThan  %[[#bool]]  %[[#arg0]] %[[#arg1]]
  ; CHECK: %[[#fselect:]] = OpSelect %[[#float_32]] %[[#fcmp]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpReturnValue %[[#fselect]]
  %0 = fcmp ogt float %x, %y
  %1 = select i1 %0, float %x, float %y
  ret float %1
}

define spir_func <4 x float> @opselect_float4_vec_test(<4 x float>  %x, <4 x float>  %y) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#fcmp:]] = OpFOrdGreaterThan  %[[#vec_4_bool]]  %[[#arg0]] %[[#arg1]]
  ; CHECK: %[[#fselect:]] = OpSelect %[[#vec4_float_32]] %[[#fcmp]] %[[#arg0]] %[[#arg1]]
  ; CHECK: OpReturnValue %[[#fselect]]
  %0 = fcmp ogt <4 x float> %x, %y
  %1 = select <4 x i1> %0, <4 x float> %x, <4 x float> %y
  ret <4 x float>  %1
}
