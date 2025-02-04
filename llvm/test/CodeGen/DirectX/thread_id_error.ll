; RUN: not opt -S -dxil-op-lower  %s 2>&1 | FileCheck %s

; DXIL operation not valid for library stage
; CHECK: in function test_thread_id
; CHECK-SAME: Cannot create ThreadId operation: Invalid stage

target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define i32 @test_thread_id(i32 %a) #0 {
entry:
  %0 = call i32 @llvm.dx.thread.id(i32 %a)
  ret i32 %0
}
