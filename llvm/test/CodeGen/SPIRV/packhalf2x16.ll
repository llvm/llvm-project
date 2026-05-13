; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[SET:%.*]] = OpExtInstImport "GLSL.std.450"
; CHECK-DAG: [[FLOAT:%.*]] = OpTypeFloat 32
; CHECK-DAG: [[FLOAT2:%.*]] = OpTypeVector [[FLOAT]] 2
; CHECK-DAG: [[UINT:%.*]] = OpTypeInt 32 0

; CHECK: [[P0:%.*]] = OpFunctionParameter [[FLOAT2]]
; CHECK: [[PACK:%.*]] = OpExtInst [[UINT]] [[SET]] PackHalf2x16 [[P0]]
; CHECK: OpReturnValue [[PACK]]
define hidden spir_func noundef i32 @_Z9test_funcj(<2 x float> noundef %0) local_unnamed_addr #0 {
  %2 = tail call i32 @llvm.spv.packhalf2x16.v2f32(<2 x float> %0)
  ret i32 %2
}
