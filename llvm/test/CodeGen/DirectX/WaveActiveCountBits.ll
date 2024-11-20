; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define void @main(i1 %expr) {
entry:
; CHECK: call i32 @dx.op.waveAllOp(i32 135, i1 %expr)
  %0 = call i32 @llvm.dx.wave.active.countbits(i1 %expr)
  ret void
}

declare i32 @llvm.dx.wave.active.countbits(i1)
