; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env spv1.4 %}

; Test lowering to spir-v backend for various types and scalar/vector

; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#uint64:]] = OpTypeInt 64 0
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_uint
; CHECK:   %[[#iexpr:]] = OpFunctionParameter %[[#uint]]
define i32 @test_uint(i32 %iexpr) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformBitwiseOr %[[#uint]] %[[#scope]] Reduce %[[#iexpr]]
  %0 = call i32 @llvm.spv.wave.reduce.or.i32(i32 %iexpr)
  ret i32 %0
}

declare i32 @llvm.spv.wave.reduce.or.i32(i32)

; CHECK-LABEL: Begin function test_uint64
; CHECK:   %[[#iexpr64:]] = OpFunctionParameter %[[#uint64]]
define i64 @test_uint64(i64 %iexpr64) {
entry:
; CHECK:   %[[#iret:]] = OpGroupNonUniformBitwiseOr %[[#uint64]] %[[#scope]] Reduce %[[#iexpr64]]
  %0 = call i64 @llvm.spv.wave.reduce.or.i64(i64 %iexpr64)
  ret i64 %0
}

declare i64 @llvm.spv.wave.reduce.or.i64(i64)
