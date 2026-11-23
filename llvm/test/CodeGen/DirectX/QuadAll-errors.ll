; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.6-compute \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=VERSION
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-vertex \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-geometry \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-hull \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-domain \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=STAGE
; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-raygeneration \
; RUN:   %s 2>&1 | FileCheck %s --check-prefix=STAGE

; VERSION: Cannot create QuadVote operation: No valid overloads for DXIL version 1.6
; STAGE: Cannot create QuadVote operation: Invalid stage

define i1 @quad_all(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

declare i1 @llvm.dx.quad.all(i1)
