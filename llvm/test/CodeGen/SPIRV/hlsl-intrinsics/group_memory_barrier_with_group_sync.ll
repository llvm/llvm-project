; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpMemoryModel Logical GLSL450

define void @test_group_memory_barrier_with_group_sync() {
entry:
  ; CHECK: %[[#TY:]] = OpTypeInt 32 0
  ; CHECK-DAG: %[[#MEM_SEM:]] = OpConstant %[[#TY]] 16
  ; CHECK-DAG: %[[#EXEC_AND_MEM_SCOPE:]] = OpConstant %[[#TY]] 2
  ; CHECK: OpControlBarrier %[[#EXEC_AND_MEM_SCOPE]] %[[#EXEC_AND_MEM_SCOPE]] %[[#MEM_SEM]]
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  ret void
}
