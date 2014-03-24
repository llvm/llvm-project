; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK-LABEL: @fneg
; R600-CHECK: -PV
; SI-CHECK-LABEL: @fneg
; SI-CHECK: V_XOR_B32
define void @fneg(float addrspace(1)* %out, float %in) {
entry:
  %0 = fsub float -0.000000e+00, %in
  store float %0, float addrspace(1)* %out
  ret void
}

; R600-CHECK-LABEL: @fneg_v2
; R600-CHECK: -PV
; R600-CHECK: -PV
; SI-CHECK-LABEL: @fneg_v2
; SI-CHECK: V_XOR_B32
; SI-CHECK: V_XOR_B32
define void @fneg_v2(<2 x float> addrspace(1)* nocapture %out, <2 x float> %in) {
entry:
  %0 = fsub <2 x float> <float -0.000000e+00, float -0.000000e+00>, %in
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; R600-CHECK-LABEL: @fneg_v4
; R600-CHECK: -PV
; R600-CHECK: -T
; R600-CHECK: -PV
; R600-CHECK: -PV
; SI-CHECK-LABEL: @fneg_v4
; SI-CHECK: V_XOR_B32
; SI-CHECK: V_XOR_B32
; SI-CHECK: V_XOR_B32
; SI-CHECK: V_XOR_B32
define void @fneg_v4(<4 x float> addrspace(1)* nocapture %out, <4 x float> %in) {
entry:
  %0 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %in
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

; DAGCombiner will transform:
; (fneg (f32 bitcast (i32 a))) => (f32 bitcast (xor (i32 a), 0x80000000))
; unless the target returns true for isNegFree()

; R600-CHECK-LABEL: @fneg_free
; R600-CHECK-NOT: XOR
; R600-CHECK: -KC0[2].Z
; SI-CHECK-LABEL: @fneg_free
; XXX: We could use V_ADD_F32_e64 with the negate bit here instead.
; SI-CHECK: V_SUB_F32_e64 v{{[0-9]}}, 0.000000e+00, s{{[0-9]}}, 0, 0, 0, 0
define void @fneg_free(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = bitcast i32 %in to float
  %1 = fsub float 0.0, %0
  store float %1, float addrspace(1)* %out
  ret void
}
