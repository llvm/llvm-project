; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Argument of llvm.dx.barrier is expected to be a mask of 
; DXIL::BarrierMode values. Chose an int value for testing.

define void @test_barrier() #0 {
entry:
  ; CHECK: call void @dx.op.barrier.i32(i32 80, i32 9)
  call void @llvm.dx.barrier(i32 noundef 9)
  ret void
}
