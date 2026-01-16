; RUN: opt -S -scalarizer -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s

; The DXIL ABI requires WaveActiveBallot to return a struct of four i32s.
; The scalarizer is allowed to rebuild this aggregate, but
; the ABI-visible return type must remain the struct.

%dx.types.fouri32 = type { i32, i32, i32, i32 }

define %dx.types.fouri32 @wave_ballot_simple(i1 noundef %p1) {
entry:
  %s = call %dx.types.fouri32 @llvm.dx.wave.ballot(i1 %p1)
  ret %dx.types.fouri32 %s
}

declare %dx.types.fouri32 @llvm.dx.wave.ballot(i1)

; CHECK: define %dx.types.fouri32 @wave_ballot_simple

; The intrinsic call must remain struct-returning
; CHECK: call { i32, i32, i32, i32 } @llvm.dx.wave.ballot

; Scalarization may occur
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue
; CHECK: extractvalue
; CHECK: insertvalue

; ABI-visible return type must remain the struct
; CHECK: ret %dx.types.fouri32
