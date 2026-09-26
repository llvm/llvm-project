; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

;; The shader environment uses the same multiplication by 1.0 as the kernel one,
;; because OpFMul is a core instruction and needs no extended instruction set.

; CHECK-DAG: %[[#Float:]] = OpTypeFloat 32
; CHECK-DAG: %[[#Float4:]] = OpTypeVector %[[#Float]] 4
; CHECK-DAG: %[[#OneFloat:]] = OpConstant %[[#Float]] 1
; CHECK-DAG: %[[#OneFloat4:]] = OpConstantComposite %[[#Float4]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]] %[[#OneFloat]]

; CHECK: %[[#XFloat:]] = OpFunctionParameter %[[#Float]]
; CHECK: %[[#RFloat:]] = OpFMul %[[#Float]] %[[#XFloat]] %[[#OneFloat]]
; CHECK: OpReturnValue %[[#RFloat]]
define float @test_canonicalize_float(float %x) {
  %r = call float @llvm.canonicalize.f32(float %x)
  ret float %r
}

; CHECK: %[[#XFloat4:]] = OpFunctionParameter %[[#Float4]]
; CHECK: %[[#RFloat4:]] = OpFMul %[[#Float4]] %[[#XFloat4]] %[[#OneFloat4]]
; CHECK: OpReturnValue %[[#RFloat4]]
define <4 x float> @test_canonicalize_v4float(<4 x float> %x) {
  %r = call <4 x float> @llvm.canonicalize.v4f32(<4 x float> %x)
  ret <4 x float> %r
}

declare float @llvm.canonicalize.f32(float)
declare <4 x float> @llvm.canonicalize.v4f32(<4 x float>)
