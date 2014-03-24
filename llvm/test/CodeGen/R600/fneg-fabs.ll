; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; DAGCombiner will transform:
; (fabs (f32 bitcast (i32 a))) => (f32 bitcast (and (i32 a), 0x7FFFFFFF))
; unless isFabsFree returns true

; R600-CHECK-LABEL: @fneg_fabs_free
; R600-CHECK-NOT: AND
; R600-CHECK: |PV.{{[XYZW]}}|
; R600-CHECK: -PV
; SI-CHECK-LABEL: @fneg_fabs_free
; SI-CHECK: V_OR_B32

define void @fneg_fabs_free(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = bitcast i32 %in to float
  %1 = call float @fabs(float %0)
  %2 = fsub float -0.000000e+00, %1
  store float %2, float addrspace(1)* %out
  ret void
}

; R600-CHECK-LABEL: @fneg_fabs_v2
; R600-CHECK: |{{(PV|T[0-9])\.[XYZW]}}|
; R600-CHECK: -PV
; R600-CHECK: |{{(PV|T[0-9])\.[XYZW]}}|
; R600-CHECK: -PV
; SI-CHECK-LABEL: @fneg_fabs_v2
; SI-CHECK: V_OR_B32
; SI-CHECK: V_OR_B32
define void @fneg_fabs_v2(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.fabs.v2f32(<2 x float> %in)
  %1 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %0
  store <2 x float> %1, <2 x float> addrspace(1)* %out
  ret void
}

; SI-CHECK-LABEL: @fneg_fabs_v4
; SI-CHECK: V_OR_B32
; SI-CHECK: V_OR_B32
; SI-CHECK: V_OR_B32
; SI-CHECK: V_OR_B32
define void @fneg_fabs_v4(<4 x float> addrspace(1)* %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.fabs.v4f32(<4 x float> %in)
  %1 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %0
  store <4 x float> %1, <4 x float> addrspace(1)* %out
  ret void
}

declare float @fabs(float ) readnone
declare <2 x float> @llvm.fabs.v2f32(<2 x float> ) readnone
declare <4 x float> @llvm.fabs.v4f32(<4 x float> ) readnone
