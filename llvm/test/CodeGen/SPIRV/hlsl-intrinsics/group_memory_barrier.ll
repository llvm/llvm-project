; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK: OpMemoryModel Logical GLSL450

define void @test_group_memory_barrier() #0 {
entry:
  ; CHECK: %[[#TY:]] = OpTypeInt 32 0
  ; CHECK-DAG: %[[#MEM_SEM:]] = OpConstant %[[#TY]] 264
  ; CHECK-DAG: %[[#EXEC_AND_MEM_SCOPE:]] = OpConstant %[[#TY]] 2
  ; CHECK: OpMemoryBarrier %[[#EXEC_AND_MEM_SCOPE]] %[[#MEM_SEM]]
  call void @llvm.spv.group.memory.barrier()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
