; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main(i32 %a, i32 %b, i32 %c) {
entry:
; CHECK: call i32 @dx.op.dot4AddPacked(i32 164, i32 %a, i32 %b, i32 %c)
  %0 = call i32 @llvm.dx.dot4add.u8packed(i32 %a, i32 %b, i32 %c)
  ret void
}

declare i32 @llvm.dx.dot4add.u8packed(i32, i32, i32)
