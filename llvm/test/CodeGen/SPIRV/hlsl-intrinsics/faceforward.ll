; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; FIXME(#136344): Change --target-env to vulkan1.3 and update this test accordingly once the issue is resolved.

; Make sure SPIRV operation function calls for faceforward are lowered correctly.

; CHECK-DAG: %[[#op_ext_glsl:]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: %[[#float_16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#vec4_float_16:]] = OpTypeVector %[[#float_16]] 4
; CHECK-DAG: %[[#float_32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#vec4_float_32:]] = OpTypeVector %[[#float_32]] 4

define noundef <4 x float> @faceforward_instcombine_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  ; CHECK: %[[#]] = OpFunction %[[#vec4_float_32]] None %[[#]]
  ; CHECK: %[[#arg0:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg1:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#arg2:]] = OpFunctionParameter %[[#vec4_float_32]]
  ; CHECK: %[[#]] = OpExtInst %[[#vec4_float_32]] %[[#op_ext_glsl]] FaceForward %[[#arg0]] %[[#arg1]] %[[#arg2]]
  %spv.fdot = call float @llvm.spv.fdot.v4f32(<4 x float> %b, <4 x float> %c)
  %fcmp = fcmp olt float %spv.fdot, 0.000000e+00
  %delta = fsub <4 x float> zeroinitializer, %a
  %select = select i1 %fcmp, <4 x float> %a, <4 x float> %delta
  ret <4 x float> %select
 }

declare half @llvm.spv.faceforward.f16(half, half, half)
declare float @llvm.spv.faceforward.f32(float, float, float)

declare <4 x half> @llvm.spv.faceforward.v4f16(<4 x half>, <4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.faceforward.v4f32(<4 x float>, <4 x float>, <4 x float>)
