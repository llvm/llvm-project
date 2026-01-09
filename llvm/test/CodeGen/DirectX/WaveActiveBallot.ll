; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

%dx.types.fouri32 = type { i32, i32, i32, i32 }

define noundef %dx.types.fouri32 @wave_ballot_simple(i1 noundef %p1) {
entry:
; CHECK: call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %p1)
  %ret = call %dx.types.fouri32 @llvm.dx.wave.ballot(i1 %p1)
  ret %dx.types.fouri32 %ret
}

declare %dx.types.fouri32 @llvm.dx.wave.ballot(i1)
