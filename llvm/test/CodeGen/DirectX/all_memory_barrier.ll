; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

define void @test_all_memory_barrier() {
entry:
  ; CHECK: call void @dx.op.barrier(i32 80, i32 10)
  call void @llvm.dx.all.memory.barrier()
  ret void
}
