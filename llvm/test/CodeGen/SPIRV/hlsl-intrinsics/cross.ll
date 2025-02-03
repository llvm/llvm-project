; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Make sure SPIRV operation function calls for cross are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec3_float_16:]] = OpTypeVector %[[#float_16]] 3
; CHECK-DAG: %[[#vec3_float_32:]] = OpTypeVector %[[#float_32]] 3

define noundef <3 x half> @cross_half4(<3 x half> noundef %a, <3 x half> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec3_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Cross %[[#arg0]] %[[#arg1]]
  %hlsl.cross = call <3 x half> @llvm.spv.cross.v3f16(<3 x half> %a, <3 x half> %b)
  ret <3 x half> %hlsl.cross
}

define noundef <3 x float> @cross_float4(<3 x float> noundef %a, <3 x float> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec3_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Cross %[[#arg0]] %[[#arg1]]
  %hlsl.cross = call <3 x float> @llvm.spv.cross.v3f32(<3 x float> %a, <3 x float> %b)
  ret <3 x float> %hlsl.cross
}

declare <3 x half> @llvm.spv.cross.v3f16(<3 x half>, <3 x half>)
declare <3 x float> @llvm.spv.cross.v3f32(<3 x float>, <3 x float>)
