; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck %s

define noundef <4 x i32> @wave_ballot_simple(i1 noundef %p1) {
entry:
; CHECK: call <4 x i32> @dx.op.waveBallot.void(i32 118, i1 %p1)
  %ret = call <4 x i32> @llvm.dx.wave.ballot(i1 %p1)
  ret <4 x i32> %ret
}

declare <4 x i32> @llvm.dx.wave.ballot(i1)
