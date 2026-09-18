; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.5-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test WaveReadLaneFirst lowering for scalar, vector, and matrix types.

; CHECK: Capability Shader
; CHECK: Capability GroupNonUniformBallot

; CHECK-DAG: %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG: %[[#v2_float:]] = OpTypeVector %[[#f32]] 2
; CHECK-DAG: %[[#v4_float:]] = OpTypeVector %[[#f32]] 4
; CHECK-DAG: %[[#bool:]] = OpTypeBool
; CHECK-DAG: %[[#scope:]] = OpConstant %[[#uint]] 3

@wide_f32_5 = internal addrspace(10) global [5 x float] zeroinitializer
@wide_f32_6 = internal addrspace(10) global [6 x float] zeroinitializer
@wide_f32_12 = internal addrspace(10) global [12 x float] zeroinitializer

; CHECK-LABEL: Begin function test_float
; CHECK: %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
define float @test_float(float %fexpr) {
entry:
; CHECK: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#f32]] %[[#scope]] %[[#fexpr]]
  %0 = call float @llvm.spv.wave.readlane.first.f32(float %fexpr)
  ret float %0
}

; CHECK-LABEL: Begin function test_int
; CHECK: %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i32 @test_int(i32 %iexpr) {
entry:
; CHECK: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#uint]] %[[#scope]] %[[#iexpr]]
  %0 = call i32 @llvm.spv.wave.readlane.first.i32(i32 %iexpr)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_bool
; CHECK: %[[#bexpr:]] = OpFunctionParameter %[[#bool]]
define i1 @test_bool(i1 %bexpr) {
entry:
; CHECK: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#bool]] %[[#scope]] %[[#bexpr]]
  %0 = call i1 @llvm.spv.wave.readlane.first.i1(i1 %bexpr)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_vfloat
; CHECK: %[[#vfexpr:]] = OpFunctionParameter %[[#v4_float]]
define <4 x float> @test_vfloat(<4 x float> %vfexpr) {
entry:
; CHECK: %[[#]] = OpGroupNonUniformBroadcastFirst %[[#v4_float]] %[[#scope]] %[[#vfexpr]]
  %0 = call <4 x float> @llvm.spv.wave.readlane.first.v4f32(
      <4 x float> %vfexpr)
  ret <4 x float> %0
}

; CHECK-LABEL: Begin function test_floatv5
define void @test_floatv5() {
entry:
  %expr = load <5 x float>, ptr addrspace(10) @wide_f32_5
; CHECK: OpGroupNonUniformBroadcastFirst %[[#v4_float]] %[[#scope]]
; CHECK: OpGroupNonUniformBroadcastFirst %[[#f32]] %[[#scope]]
  %result = call <5 x float> @llvm.spv.wave.readlane.first.v5f32(
      <5 x float> %expr)
  store <5 x float> %result, ptr addrspace(10) @wide_f32_5
  ret void
}

; CHECK-LABEL: Begin function test_float2x3
define void @test_float2x3() {
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
define void @test_float3x4() {
entry:
  %expr = load <12 x float>, ptr addrspace(10) @wide_f32_12
; CHECK-COUNT-3: OpGroupNonUniformBroadcastFirst %[[#v4_float]] %[[#scope]]
  %result = call <12 x float> @llvm.spv.wave.readlane.first.v12f32(
      <12 x float> %expr)
  store <12 x float> %result, ptr addrspace(10) @wide_f32_12
  ret void
}

declare float @llvm.spv.wave.readlane.first.f32(float)
declare i32 @llvm.spv.wave.readlane.first.i32(i32)
declare i1 @llvm.spv.wave.readlane.first.i1(i1)
declare <4 x float> @llvm.spv.wave.readlane.first.v4f32(<4 x float>)
declare <5 x float> @llvm.spv.wave.readlane.first.v5f32(<5 x float>)
declare <6 x float> @llvm.spv.wave.readlane.first.v6f32(<6 x float>)
declare <12 x float> @llvm.spv.wave.readlane.first.v12f32(<12 x float>)
