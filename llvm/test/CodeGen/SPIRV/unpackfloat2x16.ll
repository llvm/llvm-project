; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[SET:%.*]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: [[UINT:%.*]] = OpTypeInt 32 0
; CHECK-DAG: [[FLOAT:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[FLOAT2:%.*]] = OpTypeVector [[FLOAT]] 2

; CHECK: [[P0:%.*]] = OpFunctionParameter [[UINT]]
; CHECK: [[UNPACK2:%.*]] = OpExtInst [[FLOAT2]] [[SET]] UnpackHalf2x16 [[P0]]
; CHECK: [[UNPACK:%.*]] = OpCompositeExtract [[FLOAT]] [[UNPACK2]] 0
; CHECK: OpReturnValue [[UNPACK]]
define hidden spir_func noundef nofpclass(nan inf) float @_Z9test_funcj(i32 noundef %0) local_unnamed_addr #0 {
  %2 = tail call reassoc nnan ninf nsz arcp afn <2 x float> @llvm.spv.unpackhalf2x16.v2f32(i32 %0)
  %3 = extractelement <2 x float> %2, i64 0
  ret float %3
}

