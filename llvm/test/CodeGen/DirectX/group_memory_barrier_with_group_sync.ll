; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

define void @test_group_memory_barrier_with_group_sync() {
entry:
  ; CHECK: call void @dx.op.barrier(i32 80, i32 9)
  call void @llvm.dx.group.memory.barrier.with.group.sync()
  ret void
}