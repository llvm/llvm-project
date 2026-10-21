; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpMemoryModel Logical GLSL450

define void @test_all_memory_barrier_with_group_sync() #0 {
entry:
  ; CHECK: %[[#TY:]] = OpTypeInt 32 0
  ; CHECK-DAG: %[[#EXEC_SCOPE:]] = OpConstant %[[#TY]] 2
  ; CHECK-DAG: %[[#MEM_SCOPE:]] = OpConstant %[[#TY]] 1
  ; CHECK-DAG: %[[#MEM_SEM:]] = OpConstant %[[#TY]] 2376
  ; CHECK: OpControlBarrier %[[#EXEC_SCOPE]] %[[#MEM_SCOPE]] %[[#MEM_SEM]]
  call void @llvm.spv.all.memory.barrier.with.group.sync()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
