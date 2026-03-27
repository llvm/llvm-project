; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.3-vulkan-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG:   %[[#uint:]] = OpTypeInt 32 0
; CHECK-DAG:   %[[#ballot_type:]] = OpTypeVector %[[#uint]] 4
; CHECK-DAG:   %[[#bool:]] = OpTypeBool
; CHECK-DAG:   %[[#scope:]] = OpConstant %[[#uint]] 3

; CHECK-LABEL: Begin function test_fun
; CHECK:   %[[#bexpr:]] = OpFunctionParameter %[[#bool]]
define i32 @test_fun(i1 %expr) {
entry:
; CHECK:   %[[#ballot:]] = OpGroupNonUniformBallot %[[#ballot_type]] %[[#scope]] %[[#bexpr]]
; CHECK:   %[[#ret:]] = OpGroupNonUniformBallotBitCount %[[#uint]] %[[#scope]] Reduce %[[#ballot]]
  %0 = call i32 @llvm.spv.wave.active.countbits(i1 %expr)
  ret i32 %0
}

declare i32 @llvm.dx.wave.active.countbits(i1)
