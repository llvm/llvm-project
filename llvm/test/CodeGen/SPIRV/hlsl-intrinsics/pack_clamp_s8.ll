; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK: [[import:%.*]] = OpExtInstImport "GLSL.std.450"

; CHECK-DAG: [[int8:%.*]] = OpTypeInt 8 0
; CHECK-DAG: [[int8x4:%.*]] = OpTypeVector [[int8]] 4
; CHECK-DAG: [[int16:%.*]] = OpTypeInt 16 0
; CHECK-DAG: [[int16x4:%.*]] = OpTypeVector [[int16]] 4
; CHECK-DAG: [[int32:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[int32x4:%.*]] = OpTypeVector [[int32]] 4

; 65408 is -128 for SClamp
; CHECK-DAG: [[int16_65408:%.*]] = OpConstant [[int16]] 65408
; CHECK-DAG: [[int16_65408x4:%.*]] = OpConstantComposite [[int16x4]] [[int16_65408]] [[int16_65408]] [[int16_65408]] [[int16_65408]]
; CHECK-DAG: [[int16_127:%.*]] = OpConstant [[int16]] 127
; CHECK-DAG: [[int16_127x4:%.*]] = OpConstantComposite [[int16x4]] [[int16_127]] [[int16_127]] [[int16_127]] [[int16_127]]

; 4294967168 is -128 for SClamp
; CHECK-DAG: [[int32_4294967168:%.*]] = OpConstant [[int32]] 4294967168
; CHECK-DAG: [[int32_4294967168x4:%.*]] = OpConstantComposite [[int32x4]] [[int32_4294967168]] [[int32_4294967168]] [[int32_4294967168]] [[int32_4294967168]]
; CHECK-DAG: [[int32_127:%.*]] = OpConstant [[int32]] 127
; CHECK-DAG: [[int32_127x4:%.*]] = OpConstantComposite [[int32x4]] [[int32_127]] [[int32_127]] [[int32_127]] [[int32_127]]

define noundef i32 @pack_clamp_s8_16(<4 x i16> noundef %a) {
; CHECK: [[in:%.*]] = OpFunctionParameter [[int16x4]]
; CHECK: [[clamped:%.*]] = OpExtInst [[int16x4]] [[import]] SClamp [[in]] [[int16_65408x4]] [[int16_127x4]]
; CHECK: [[converted:%.*]] = OpSConvert [[int8x4]] [[clamped]]
; CHECK: [[cast:%.*]] = OpBitcast [[int32]] [[converted]]
  %packed = call i32 @llvm.spv.pack.clamp.s8.v4i16(<4 x i16> %a)
  ret i32 %packed
}

define noundef i32 @pack_clamp_s8_32(<4 x i32> noundef %a) {
; CHECK: [[in:%.*]] = OpFunctionParameter [[int32x4]]
; CHECK: [[clamped:%.*]] = OpExtInst [[int32x4]] [[import]] SClamp [[in]] [[int32_4294967168x4]] [[int32_127x4]]
; CHECK: [[converted:%.*]] = OpSConvert [[int8x4]] [[clamped]]
; CHECK: [[cast:%.*]] = OpBitcast [[int32]] [[converted]]
  %packed = call i32 @llvm.spv.pack.clamp.s8.v4i32(<4 x i32> %a)
  ret i32 %packed
}

declare i32 @llvm.spv.pack.clamp.s8.v4i16(<4 x i16>)
declare i32 @llvm.spv.pack.clamp.s8.v4i32(<4 x i32>)
