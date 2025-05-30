; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s 2>&1 | FileCheck %s

; CHECK: in function f
; CHECK-SAME: Cannot create Dot4AddU8Packed operation: No valid overloads for DXIL version 1.3

define void @f(i32 %acc, i32 %x, i32 %y) {
entry:
  %0 = call i32 @llvm.dx.dot4add.u8packed(i32 %acc, i32 %x, i32 %y)
  ret void
}
