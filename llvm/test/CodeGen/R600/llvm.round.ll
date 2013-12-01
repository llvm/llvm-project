; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600 --check-prefix=FUNC

; FUNC-LABEL: @f32
; R600: FRACT
; R600-DAG: ADD
; R600-DAG: CEIL
; R600-DAG: FLOOR
; R600: CNDGE
define void @f32(float addrspace(1)* %out, float %in) {
entry:
  %0 = call float @llvm.round.f32(float %in)
  store float %0, float addrspace(1)* %out
  ret void
}

; The vector tests are really difficult to verify, since it can be hard to
; predict how the scheduler will order the instructions.  We already have
; a test for the scalar case, so the vector tests just check that the
; compiler doesn't crash.

; FUNC-LABEL: v2f32
; R600: CF_END
define void @v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.round.v2f32(<2 x float> %in)
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: v4f32
; R600: CF_END
define void @v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.round.v4f32(<4 x float> %in)
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

declare float @llvm.round.f32(float)
declare <2 x float> @llvm.round.v2f32(<2 x float>)
declare <4 x float> @llvm.round.v4f32(<4 x float>)
