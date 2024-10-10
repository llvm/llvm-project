; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend for various types and scalar/vector

; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:   %[[#v4_bool:]] = OpTypeVector %[[#bool]] 4
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
; CHECK:   %[[#idx1:]] = OpFunctionParameter %[[#uint]]
define float @test_1(float %fexpr, i32 %idx) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformShuffle %[[#f32]] %[[#fexpr]] %[[#idx1]] %[[#scope]]
  %0 = call float @llvm.spv.wave.readlaneat.f32(float %fexpr, i32 %idx)
  ret float %0
}

; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
; CHECK:   %[[#idx2:]] = OpFunctionParameter %[[#uint]]
define i32 @test_2(i32 %iexpr, i32 %idx) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformShuffle %[[#uint]] %[[#iexpr]] %[[#idx2]] %[[#scope]]
  %0 = call i32 @llvm.spv.wave.readlaneat.i32(i32 %iexpr, i32 %idx)
  ret i32 %0
}

; CHECK:   %[[#vbexpr:]] = OpFunctionParameter %[[#v4_bool]]
; CHECK:   %[[#idx3:]] = OpFunctionParameter %[[#uint]]
define <4 x i1> @test_3(<4 x i1> %vbexpr, i32 %idx) {
entry:
; CHECK:   %[[#vbret:]] = OpGroupNonUniformShuffle %[[#v4_bool]] %[[#vbexpr]] %[[#idx3]] %[[#scope]]
  %0 = call <4 x i1> @llvm.spv.wave.readlaneat.v4i1(<4 x i1> %vbexpr, i32 %idx)
  ret <4 x i1> %0
}

declare float @llvm.spv.wave.readlaneat.f32(float, i32)
declare i32 @llvm.spv.wave.readlaneat.i32(i32, i32)
declare <4 x i1> @llvm.spv.wave.readlaneat.v4i1(<4 x i1>, i32)
