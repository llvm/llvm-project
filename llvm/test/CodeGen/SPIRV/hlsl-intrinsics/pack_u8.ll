; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[int8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[int8x4:%.*]] = OpTypeVector [[int8]] 4
; CHECK-DAG: [[int16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[int16x4:%.*]] = OpTypeVector [[int16]] 4
; CHECK-DAG: [[int32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[int32x4:%.*]] = OpTypeVector [[int32]] 4

define noundef i32 @pack_u8_16(<4 x i16> noundef %a) {
; CHECK: [[in:%.*]] = OpFunctionParameter [[int16x4]]
; CHECK: [[converted:%.*]] = OpUConvert [[int8x4]] [[in]]
; CHECK: [[cast:%.*]] = OpBitcast [[int32]] [[converted]]
  %packed = call i32 @llvm.spv.pack.u8.v4i16(<4 x i16> %a)
  ret i32 %packed
}

define noundef i32 @pack_u8_32(<4 x i32> noundef %a) {
; CHECK: [[in:%.*]] = OpFunctionParameter [[int32x4]]
; CHECK: [[converted:%.*]] = OpUConvert [[int8x4]] [[in]]
; CHECK: [[cast:%.*]] = OpBitcast [[int32]] [[converted]]
  %packed = call i32 @llvm.spv.pack.u8.v4i32(<4 x i32> %a)
  ret i32 %packed
}

declare i32 @llvm.spv.pack.u8.v4i16(<4 x i16>)
declare i32 @llvm.spv.pack.u8.v4i32(<4 x i32>)
