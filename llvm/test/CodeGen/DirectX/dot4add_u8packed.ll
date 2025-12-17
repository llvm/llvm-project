; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.4-compute %s | FileCheck %s

define void @main(i32 %acc, i32 %x, i32 %y) {
entry:
; CHECK: call i32 @dx.op.dot4AddPacked.i32(i32 164, i32 %acc, i32 %x, i32 %y) #[[#ATTR:]]
  %0 = call i32 @llvm.dx.dot4add.u8packed(i32 %acc, i32 %x, i32 %y)
  ret void
}

; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}

declare i32 @llvm.dx.dot4add.u8packed(i32, i32, i32)
