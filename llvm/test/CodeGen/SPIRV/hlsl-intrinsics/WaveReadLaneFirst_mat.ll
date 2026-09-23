; XFAIL: *
; TODO: Support matrix legalization for SPIR-V target intrinsics.
; https://github.com/llvm/llvm-project/issues/225961
;
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Test WaveReadLaneFirst lowering for matrix types without long vectors.

; CHECK: Capability Shader
; CHECK: Capability GroupNonUniformBallot
; CHECK-NOT: Capability LongVectorEXT
; CHECK-NOT: Extension "SPV_EXT_long_vector"

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v2_float:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#v4_float:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3

@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f32_12 = internal addrspace(10) global [12 x float] zeroinitializer

; CHECK-LABEL: Begin function test_float2x3
define internal void @test_float2x3() {
entry:
  %expr = load <6 x float>, ptr addrspace(10) @wide_f32_6
; CHECK: OpGroupNonUniformBroadcastFirst %[[#v4_float]] %[[#scope]]
; CHECK: OpGroupNonUniformBroadcastFirst %[[#v2_float]] %[[#scope]]
  %result = call <6 x float> @llvm.spv.wave.readlane.first.v6f32(
      <6 x float> %expr)
  store <6 x float> %result, ptr addrspace(10) @wide_f32_6
  ret void
}

; CHECK-LABEL: Begin function test_float3x4
define internal void @test_float3x4() {
entry:
  %expr = load <12 x float>, ptr addrspace(10) @wide_f32_12
; CHECK-COUNT-3: OpGroupNonUniformBroadcastFirst %[[#v4_float]] %[[#scope]]
  %result = call <12 x float> @llvm.spv.wave.readlane.first.v12f32(
      <12 x float> %expr)
  store <12 x float> %result, ptr addrspace(10) @wide_f32_12
  ret void
}

define void @main() #0 {
  ret void
}

declare <6 x float> @llvm.spv.wave.readlane.first.v6f32(<6 x float>)
declare <12 x float> @llvm.spv.wave.readlane.first.v12f32(<12 x float>)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
