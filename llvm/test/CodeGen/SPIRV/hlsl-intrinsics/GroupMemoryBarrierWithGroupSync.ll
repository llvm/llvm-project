; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

define void @test_group_memory_barrier_with_group_sync() {
entry:
  ; CHECK: call void @dx.op.barrier(i32 80, i32 9)
  call void @llvm.spv.groupMemoryBarrierWithGroupSync()
  ret void
}
