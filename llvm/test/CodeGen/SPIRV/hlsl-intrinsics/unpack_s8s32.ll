; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[int8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[int8x4:%.*]] = OpTypeVector [[int8]] 4
; CHECK-DAG: [[int32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[int32x4:%.*]] = OpTypeVector [[int32]] 4

define noundef <4 x i32> @unpack_s8s32(i32 noundef %a) {
; CHECK: [[in:%.*]] = OpFunctionParameter
; CHECK: [[cast:%.*]] = OpBitcast [[int8x4]] [[in]]
; CHECK: [[converted:%.*]] = OpSConvert [[int32x4]] [[cast]]
  %unpacked = call <4 x i32> @llvm.spv.unpack.s8s32(i32 %a)
  ret <4 x i32> %unpacked
}

declare <4 x i32> @llvm.spv.unpack.s8s32(i32)
