; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define noundef i1 @wave_any_simple(i1 noundef %p1) {
entry:
; CHECK: call i1 @dx.op.waveAnyTrue(i32 113, i1 %p1){{$}}
  %ret = call i1 @llvm.dx.wave.any(i1 %p1)
  ret i1 %ret
}

declare i1 @llvm.dx.wave.any(i1)
