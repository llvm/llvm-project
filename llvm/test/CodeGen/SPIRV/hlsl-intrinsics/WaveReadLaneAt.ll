; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend

; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:   %[[#v4_bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
; CHECK-DAG:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
; CHECK-DAG:   %[[#idx:]] = OpFunctionParameter %[[#uint]]
; CHECK-DAG:   %[[#vbexpr:]] = OpFunctionParameter %[[#v4_bool]]

define spir_func void @test_1(float %fexpr, i32 %iexpr, <4 x i1> %vbexpr, i32 %idx) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformShuffle %[[#f32]] %[[#fexpr]] %[[#idx]] %[[#scope]]
  %0 = call float @llvm.spv.wave.read.lane.at.f32(float %fexpr, i32 %idx)
; CHECK:   %[[#iret:]] = OpGroupNonUniformShuffle %[[#uint]] %[[#iexpr]] %[[#idx]] %[[#scope]]
  %1 = call i32 @llvm.spv.wave.read.lane.at.i32(i32 %iexpr, i32 %idx)
; CHECK:   %[[#vbret:]] = OpGroupNonUniformShuffle %[[#v4_bool]] %[[#vbexpr]] %[[#idx]] %[[#scope]]
  %2 = call <4 x i1> @llvm.spv.wave.read.lane.at.v4i1(<4 x i1> %vbexpr, i32 %idx)
  ret void
}

declare float @__hlsl_wave_read_lane_at.f32(float, i32)
declare i32 @__hlsl_wave_read_lane_at.i32(i32, i32)
declare <4 x i1> @__hlsl_wave_read_lane_at.v4i1(<4 x i1>, i32)
