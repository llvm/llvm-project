; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-compute %s | FileCheck --check-prefix LOWER %s
; RUN: opt -S -scalarizer -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck --check-prefix SCALAR %s

%dx.types.fouri32 = type { i32, i32, i32, i32 }

define %dx.types.fouri32 @wave_ballot_simple(i1 noundef %p1) {
entry:
; LOWER: call %dx.types.fouri32 @dx.op.waveActiveBallot(i32 116, i1 %p1)
; SCALAR: call { i32, i32, i32, i32 } @llvm.dx.wave.ballot.i32(i1 %p1)

  %s = call %dx.types.fouri32 @llvm.dx.wave.ballot(i1 %p1)
 
; Scalarization may occur
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue

; CHECK-NOT: ret %dx.types.fouri32
; CHECK: ret <4 x i32>
  ret %dx.types.fouri32 %s
}

declare %dx.types.fouri32 @llvm.dx.wave.ballot(i1)
