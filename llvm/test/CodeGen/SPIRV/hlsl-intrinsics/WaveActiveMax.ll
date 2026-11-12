; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend for various types and scalar/vector

; CHECK: OpCapability GroupNonUniformArithmetic

; CHECK-DAG:   %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#v4_half:]] = OpTypeVector %[[#f16]] 4
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_float
; CHECK:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
define float @test_float(float %fexpr) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformFMax %[[#f32]] %[[#scope]] Reduce %[[#fexpr]]
  %0 = call float @llvm.spv.wave.reduce.max.f32(float %fexpr)
  ret float %0
}

; CHECK-LABEL: Begin function test_int_signed
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i32 @test_int_signed(i32 %iexpr) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformSMax %[[#uint]] %[[#scope]] Reduce %[[#iexpr]]
  %0 = call i32 @llvm.spv.wave.reduce.max.i32(i32 %iexpr)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_int_unsigned
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i32 @test_int_unsigned(i32 %iexpr) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformUMax %[[#uint]] %[[#scope]] Reduce %[[#iexpr]]
  %0 = call i32 @llvm.spv.wave.reduce.umax.i32(i32 %iexpr)
  ret i32 %0
}

; CHECK-LABEL: Begin function test_vhalf
; CHECK:   %[[#vbexpr:]] = OpFunctionParameter %[[#v4_half]]
define <4 x half> @test_vhalf(<4 x half> %vbexpr) {
entry:
; CHECK:   %[[#vhalfret:]] = OpGroupNonUniformFMax %[[#v4_half]] %[[#scope]] Reduce %[[#vbexpr]]
  %0 = call <4 x half> @llvm.spv.wave.reduce.max.v4half(<4 x half> %vbexpr)
  ret <4 x half> %0
}

declare float @llvm.spv.wave.reduce.max.f32(float)
declare i32 @llvm.spv.wave.reduce.max.i32(i32)
declare <4 x half> @llvm.spv.wave.reduce.max.v4half(<4 x half>)

declare float @llvm.spv.wave.reduce.umax.f32(float)
declare i32 @llvm.spv.wave.reduce.umax.i32(i32)
declare <4 x half> @llvm.spv.wave.reduce.umax.v4half(<4 x half>)

