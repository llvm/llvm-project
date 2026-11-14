; RUN: split-file %s %t
; RUN: not opt -S -dxil-op-lower %t/pixel.ll 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower %t/sm67.ll 2>&1 | FileCheck %s --check-prefix=VERSION

; StartVertexLocation is only available to vertex shaders from shader model 6.8.
; STAGE: Cannot create StartVertexLocation operation: Invalid stage
; VERSION: Cannot create StartVertexLocation operation: No valid overloads for DXIL version 1.7

;--- pixel.ll
target triple = "dxil-pc-shadermodel6.8-pixel"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.start.vertex.location()
  ret i32 %0
}

;--- sm67.ll
target triple = "dxil-pc-shadermodel6.7-vertex"
define i32 @main() {
entry:
  %0 = call i32 @llvm.dx.start.vertex.location()
  ret i32 %0
}
