; RUN: not opt -S -dxil-op-lower  %s 2>&1 | FileCheck %s

; DXIL operation not valid for library stage
; CHECK: LLVM ERROR: library : Invalid Shader Stage for DXIL operation - ThreadId

target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-pc-shadermodel6.7-library"

; Function Attrs: noinline nounwind optnone
define i32 @test_thread_id(i32 %a) #0 {
entry:
  %0 = call i32 @llvm.dx.thread.id(i32 %a)
  ret i32 %0
}
