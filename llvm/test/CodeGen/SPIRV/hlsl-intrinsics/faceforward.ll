; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-vulkan1.3-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan1.3-unknown %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Make sure SPIRV operation function calls for faceforward are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define internal noundef half @faceforward_half(half noundef %a, half noundef %b, half noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_16]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.faceforward = call half @llvm.spv.faceforward.f16(half %a, half %b, half %c)
  ret half %spv.faceforward
}

define internal noundef float @faceforward_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.faceforward = call float @llvm.spv.faceforward.f32(float %a, float %b, float %c)
  ret float %spv.faceforward
}

define internal noundef <4 x half> @faceforward_half4(<4 x half> noundef %a, <4 x half> noundef %b, <4 x half> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_16]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_16]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_16]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.faceforward = call <4 x half> @llvm.spv.faceforward.v4f16(<4 x half> %a, <4 x half> %b, <4 x half> %c)
  ret <4 x half> %spv.faceforward
}

define internal noundef <4 x float> @faceforward_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.faceforward = call <4 x float> @llvm.spv.faceforward.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %spv.faceforward
}

define internal noundef float @faceforward_instcombine_float(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %fmul= fmul float %b, %c
  %fcmp = fcmp olt float %fmul, 0.000000e+00
  %fneg = fneg float %a
  %select = select i1 %fcmp, float %a, float %fneg
  ret float %select
 }

define internal noundef <4 x float> @faceforward_instcombine_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp olt float %spv.fdot, 0.000000e+00
  %fneg = fneg <4 x float> %a
  %select = select i1 %fcmp, <4 x float> %fneg, <4 x float> %a
  ret <4 x float> %select
 }

define internal noundef float @faceforward_instcombine_float_constants(float noundef %a, float noundef %b, float noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#float_32]] %[[#op_ext_glsl]] FaceForward %[[#]] %[[#arg1]] %[[#arg2]]
  %fmul = fmul float %b, %c
  %fcmp = fcmp olt float %fmul, -0.000000e+00
  %select = select i1 %fcmp, float 1.000000e+00, float -1.000000e+00
  ret float %select
}

define internal noundef <4 x float> @faceforward_instcombine_float4_constants(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp olt float %spv.fdot, 0.000000e+00
  %select = select i1 %fcmp, <4 x float> <float 1.000000e+00, float 2.000000e+00, float 3.000000e+00, float 4.000000e+00>, <4 x float> <float -1.000000e+00, float -2.000000e+00, float -3.000000e+00, float -4.000000e+00>
  ret <4 x float> %select
}

define internal noundef <4 x float> @faceforward_instcombine_float4_splat(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp olt float %spv.fdot, 0.000000e+00
  %select = select i1 %fcmp, <4 x float> splat (float 2.500000e+00), <4 x float> splat (float -2.500000e+00)
  ret <4 x float> %select
}

define internal noundef <4 x float> @faceforward_instcombine_float4_false_fmul(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp olt float %spv.fdot, 0.000000e+00
  %fneg = fmul <4 x float> %a, <float -1.000000e+00, float -1.000000e+00, float -1.000000e+00, float -1.000000e+00>
  %select = select i1 %fcmp, <4 x float> %a, <4 x float> %fneg
  ret <4 x float> %select
}

define internal noundef <4 x float> @faceforward_instcombine_float4_ogt(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp ogt float 0.000000e+00, %spv.fdot
  %fneg = fneg <4 x float> %a
  %select = select i1 %fcmp, <4 x float> %fneg, <4 x float> %a
  ret <4 x float> %select
}

; The other functions are the test, but a entry point is required to have a valid SPIR-V module.
define void @main() #1 {
entry:
  ret void
}

declare half @llvm.spv.faceforward.f16(half, half, half)
declare float @llvm.spv.faceforward.f32(float, float, float)

declare <4 x half> @llvm.spv.faceforward.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.faceforward.v4f32(<4 x float>, <4 x float>, <4 x float>)

attributes #1 = { convergent noinline norecurse "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
