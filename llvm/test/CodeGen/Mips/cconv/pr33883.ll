; RUN: llc -mtriple=mips -mcpu=mips32 < %s -o /dev/null

; Test that calls to vector intrinsics do not crash SelectionDAGBuilder.

define <4 x float> @_ZN4simd3foo17hebb969c5fb39a194E(<4 x float>) {
start:
  %1 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)

  ret <4 x float> %1
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)
