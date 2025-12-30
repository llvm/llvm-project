; RUN: not opt -S -dxil-op-lower  %s 2>&1 | FileCheck %s

; DXIL operation sin is not valid in library stage
; CHECK: in function test_flattened_thread_id_in_group
; CHECK-SAME: Cannot create FlattenedThreadIdInGroup operation: Invalid stage

target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define i32 @test_flattened_thread_id_in_group() #0 {
entry:
  %0 = call i32 @llvm.dx.flattened.thread.id.in.group()
  ret i32 %0
}
