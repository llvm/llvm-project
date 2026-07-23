; RUN: llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

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

; Make sure the manually expanded cross product (as emitted by the header-only
; HLSL implementation) is combined back into the Cross extended instruction.

define noundef <3 x half> @cross_instcombine_half3(<3 x half> noundef %a, <3 x half> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec3_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec3_float_16]] %[[#op_ext_glsl]] Cross %[[#arg0]] %[[#arg1]]
  %a0 = extractelement <3 x half> %a, i64 0
  %a1 = extractelement <3 x half> %a, i64 1
  %a2 = extractelement <3 x half> %a, i64 2
  %b0 = extractelement <3 x half> %b, i64 0
  %b1 = extractelement <3 x half> %b, i64 1
  %b2 = extractelement <3 x half> %b, i64 2
  %mul0 = fmul half %a1, %b2
  %mul1 = fmul half %b1, %a2
  %sub0 = fsub half %mul0, %mul1
  %vec0 = insertelement <3 x half> poison, half %sub0, i64 0
  %mul2 = fmul half %a2, %b0
  %mul3 = fmul half %b2, %a0
  %sub1 = fsub half %mul2, %mul3
  %vec1 = insertelement <3 x half> %vec0, half %sub1, i64 1
  %mul4 = fmul half %a0, %b1
  %mul5 = fmul half %b0, %a1
  %sub2 = fsub half %mul4, %mul5
  %vec2 = insertelement <3 x half> %vec1, half %sub2, i64 2
  ret <3 x half> %vec2
}

define noundef <3 x float> @cross_instcombine_float3(<3 x float> noundef %a, <3 x float> noundef %b) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec3_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec3_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec3_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec3_float_32]] %[[#op_ext_glsl]] Cross %[[#arg0]] %[[#arg1]]
  %a0 = extractelement <3 x float> %a, i64 0
  %a1 = extractelement <3 x float> %a, i64 1
  %a2 = extractelement <3 x float> %a, i64 2
  %b0 = extractelement <3 x float> %b, i64 0
  %b1 = extractelement <3 x float> %b, i64 1
  %b2 = extractelement <3 x float> %b, i64 2
  %mul0 = fmul float %a1, %b2
  %mul1 = fmul float %b1, %a2
  %sub0 = fsub float %mul0, %mul1
  %vec0 = insertelement <3 x float> poison, float %sub0, i64 0
  %mul2 = fmul float %a2, %b0
  %mul3 = fmul float %b2, %a0
  %sub1 = fsub float %mul2, %mul3
  %vec1 = insertelement <3 x float> %vec0, float %sub1, i64 1
  %mul4 = fmul float %a0, %b1
  %mul5 = fmul float %b0, %a1
  %sub2 = fsub float %mul4, %mul5
  %vec2 = insertelement <3 x float> %vec1, float %sub2, i64 2
  ret <3 x float> %vec2
}

declare <3 x half> @llvm.spv.cross.v3f16(<3 x half>, <3 x half>)
declare <3 x float> @llvm.spv.cross.v3f32(<3 x float>, <3 x float>)
