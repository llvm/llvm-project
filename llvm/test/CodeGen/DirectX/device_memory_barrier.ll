; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

define void @test_device_memory_barrier() {
entry:
  ; CHECK: call void @dx.op.barrier(i32 80, i32 2)
  call void @llvm.dx.device.memory.barrier()
  ret void
}
