; RUN: not opt -S -dxil-op-lower  %s 2>&1 | FileCheck %s

; DXIL operation sin is not valid in vertex stage
; CHECK: in function test_thread_id_in_group
; CHECK-SAME: Cannot create ThreadIdInGroup operation: Invalid stage

target triple = "dxil-pc-shadermodel6.7-vertex"

; Function Attrs: noinline nounwind optnone
define i32 @test_thread_id_in_group(i32 %a) #0 {
entry:
  %0 = call i32 @llvm.dx.thread.id.in.group(i32 %a)
  ret i32 %0
}
