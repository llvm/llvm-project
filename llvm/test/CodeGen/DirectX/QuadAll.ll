; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-library %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-compute %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-pixel %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-amplification %s \
; RUN:   | FileCheck %s
; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.7-mesh %s \
; RUN:   | FileCheck %s

; CHECK-LABEL: define i1 @quad_all(
; CHECK: %[[RET:.*]] = call i1 @dx.op.quadVote.i1(i32 222, i1 %cond, i8 1)
; CHECK-NEXT: ret i1 %[[RET]]
define i1 @quad_all(i1 %cond) {
  %ret = call i1 @llvm.dx.quad.all(i1 %cond)
  ret i1 %ret
}

declare i1 @llvm.dx.quad.all(i1)
