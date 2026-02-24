; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; Test lowering to spir-v backend for various types and scalar/vector

; CHECK-DAG:   %[[#f16:]] = OpTypeFloat 16
; CHECK-DAG:   %[[#f32:]] = OpTypeFloat 32
; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#v4_half:]] = OpTypeVector %[[#f16]] 4
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_float
; CHECK:   %[[#fexpr:]] = OpFunctionParameter %[[#f32]]
define i1 @test_float(float %fexpr) {
entry:
; CHECK:   %[[#fret:]] = OpGroupNonUniformAllEqual %[[#f32]] %[[#scope]] Reduce %[[#fexpr]]
  %0 = call i1 @llvm.spv.wave.all.equal.f32(float %fexpr)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_int
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i1 @test_int(i32 %iexpr) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformAllEqual %[[#uint]] %[[#scope]] Reduce %[[#iexpr]]
  %0 = call i1 @llvm.spv.wave.all.equal.i32(i32 %iexpr)
  ret i1 %0
}

; CHECK-LABEL: Begin function test_vhalf
; CHECK:   %[[#vbexpr:]] = OpFunctionParameter %[[#v4_half]]
define i1 @test_vhalf(<4 x half> %vbexpr) {
entry:
; CHECK:   %[[#vhalfret:]] = OpGroupNonUniformAllEqual %[[#v4_half]] %[[#scope]] Reduce %[[#vbexpr]]
  %0 = call i1 @llvm.spv.wave.all.equal.v4half(<4 x half> %vbexpr)
  ret i1 %0
}

declare i1 @llvm.spv.wave.all.equal.f32(float)
declare i1 @llvm.spv.wave.all.equal.i32(i32)
declare i1 @llvm.spv.wave.all.equal.v4half(<4 x half>)
