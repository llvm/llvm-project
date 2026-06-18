; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#vec4_bool:]] = OpTypeVector %[[#bool]] 4

define noundef i1 @isnormal_half(half noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#bool]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#]] = OpIsNormal %[[#bool]] %[[#arg0]]
  %hlsl.isnormal = call i1 @llvm.spv.isnormal.f16(half %a)
  ret i1 %hlsl.isnormal
}

define noundef i1 @isnormal_float(float noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#bool]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpIsNormal %[[#bool]] %[[#arg0]]
  %hlsl.isnormal = call i1 @llvm.spv.isnormal.f32(float %a)
  ret i1 %hlsl.isnormal
}

define noundef <4 x i1> @isnormal_half4(<4 x half> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_bool]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpIsNormal %[[#vec4_bool]] %[[#arg0]]
  %hlsl.isnormal = call <4 x i1> @llvm.spv.isnormal.v4f16(<4 x half> %a)
  ret <4 x i1> %hlsl.isnormal
}

define noundef <4 x i1> @isnormal_float4(<4 x float> noundef %a) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_bool]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpIsNormal %[[#vec4_bool]] %[[#arg0]]
  %hlsl.isnormal = call <4 x i1> @llvm.spv.isnormal.v4f32(<4 x float> %a)
  ret <4 x i1> %hlsl.isnormal
}
