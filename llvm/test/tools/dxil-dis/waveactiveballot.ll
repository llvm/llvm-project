; RUN: llc %s --filetype=obj -o - | dxil-dis -o - | FileCheck %s

; CHECK-NOT: llvm.dx.wave.ballot

; CHECK: call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %p1)
; CHECK-NOT: ret %dx.types.fouri32
; CHECK: ret <4 x i32>


target triple = "dxil-unknown-shadermodel6.3-library"

%dx.types.fouri32 = type { i32, i32, i32, i32 }

define <4 x i32> @wave_ballot_simple(i1 %p1) {
entry:
  %s = call %dx.types.fouri32 @llvm.dx.wave.ballot(i1 %p1)

  %v0 = extractvalue %dx.types.fouri32 %s, 0
  %v1 = extractvalue %dx.types.fouri32 %s, 1
  %v2 = extractvalue %dx.types.fouri32 %s, 2
  %v3 = extractvalue %dx.types.fouri32 %s, 3

  %vec0 = insertelement <4 x i32> poison, i32 %v0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %v1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %v2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %v3, i32 3

  ret <4 x i32> %vec3
}

declare %dx.types.fouri32 @llvm.dx.wave.ballot(i1)
