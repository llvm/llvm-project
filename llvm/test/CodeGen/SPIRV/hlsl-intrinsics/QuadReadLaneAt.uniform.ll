; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv1.5-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend for various types and scalar/vector
; This tests SPIRV 1.5 where index must be dynamically uniform

; CHECK: OpCapability GroupNonUniformQuad

; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:   %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#v4_half:]] = OpTypeVector %[[#f16]] 4
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_bool
; CHECK:   %[[#bexpr:]] = OpFunctionParameter %[[#bool]]
; CHECK:   %[[#idx:]] = OpFunctionParameter %[[#uint]]
define internal i1 @test_bool(i1 %bexpr, i32 %idx) {
entry:
; CHECK:   %[[#bret:]] = OpGroupNonUniformQuadBroadcast %[[#bool]] %[[#scope]] %[[#bexpr]] %[[#idx]]
  %0 = call i1 @llvm.spv.quad.read.lane.at.i1(i1 %bexpr, i32 %idx)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_float
; CHECK:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
; CHECK:   %[[#idx:]] = OpFunctionParameter %[[#uint]]
define internal float @test_float(float %fexpr, i32 %idx) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformQuadBroadcast %[[#f32]] %[[#scope]] %[[#fexpr]] %[[#idx]]
  %0 = call float @llvm.spv.quad.read.lane.at.f32(float %fexpr, i32 %idx)
  ret float %0
}

; CHECK-LABEL: Begin function test_int
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
; CHECK:   %[[#idx:]] = OpFunctionParameter %[[#uint]]
define internal i32 @test_int(i32 %iexpr, i32 %idx) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformQuadBroadcast %[[#uint]] %[[#scope]] %[[#iexpr]] %[[#idx]]
  %0 = call i32 @llvm.spv.quad.read.lane.at.i32(i32 %iexpr, i32 %idx)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_vhalf
; CHECK:   %[[#vbexpr:]] = OpFunctionParameter %[[#v4_half]]
; CHECK:   %[[#idx:]] = OpFunctionParameter %[[#uint]]
define internal <4 x half> @test_vhalf(<4 x half> %vbexpr, i32 %idx) {
entry:
; CHECK:   %[[#vhalfret:]] = OpGroupNonUniformQuadBroadcast %[[#v4_half]] %[[#scope]] %[[#vbexpr]] %[[#idx]]
  %0 = call <4 x half> @llvm.spv.quad.read.lane.at.v4half(<4 x half> %vbexpr, i32 %idx)
  ret <4 x half> %0
}

define void @main() #0 {
  ret void
}

declare i1 @llvm.spv.quad.read.lane.at.i1(i1, i32)
declare float @llvm.spv.quad.read.lane.at.f32(float, i32)
declare i32 @llvm.spv.quad.read.lane.at.i32(i32, i32)
declare <4 x half> @llvm.spv.quad.read.lane.at.v4half(<4 x half>, i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
