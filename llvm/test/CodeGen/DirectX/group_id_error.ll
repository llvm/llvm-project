; RUN: not opt -S -dxil-op-lower  %s 2>&1 | FileCheck %s

; DXIL operation not valid for pixel stage
; CHECK: LLVM ERROR: pixel : Invalid Shader Stage for DXIL operation - GroupId

target triple = "dxil-pc-shadermodel6.7-pixel"

; Function Attrs: noinline nounwind optnone
define i32 @test_group_id(i32 %a) #0 {
entry:
  %0 = call i32 @llvm.dx.group.id(i32 %a)
  ret i32 %0
}
