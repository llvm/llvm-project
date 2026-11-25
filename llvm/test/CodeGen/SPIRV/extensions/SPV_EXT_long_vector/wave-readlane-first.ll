; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_EXT_long_vector %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute --spirv-ext=+SPV_EXT_long_vector %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; Test that SPV_EXT_long_vector preserves a WaveReadLaneFirst long vector.

; CHECK-DAG: Capability Shader
; CHECK-DAG: Capability GroupNonUniformBallot
; CHECK-DAG: Capability LongVectorEXT
; CHECK-DAG: Extension "SPV_EXT_long_vector"

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3
; CHECK-DAG: %[[#size5:]] = OpConstant %[[#uint]] 5
; CHECK-DAG: %[[#v5_float:]] = OpTypeVectorIdEXT %[[#f32]] %[[#size5]]

@wide_f32_5 = internal addrspace(10) global [5 x float] zeroinitializer

; CHECK-LABEL: Begin function test_floatv5
define internal void @test_floatv5() {
entry:
  %expr = load <5 x float>, ptr addrspace(10) @wide_f32_5
; CHECK: OpGroupNonUniformBroadcastFirst %[[#v5_float]] %[[#scope]]
  %result = call <5 x float> @llvm.spv.wave.readlane.first.v5f32(
      <5 x float> %expr)
  store <5 x float> %result, ptr addrspace(10) @wide_f32_5
  ret void
}

define void @main() #0 {
  ret void
}

declare <5 x float> @llvm.spv.wave.readlane.first.v5f32(<5 x float>)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
