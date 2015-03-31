; RUN: llc < %s -march=r600 --mcpu=redwood | FileCheck %s --check-prefix=R600-CHECK
; RUN: llc < %s -march=r600 --mcpu=SI -verify-machineinstrs| FileCheck %s --check-prefix=SI-CHECK
; RUN: llc < %s -march=r600 --mcpu=tonga -verify-machineinstrs| FileCheck %s --check-prefix=SI-CHECK

; R600-CHECK-LABEL: {{^}}sqrt_f32:
; R600-CHECK: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[2].Z
; R600-CHECK: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[2].Z, PS
; SI-CHECK-LABEL: {{^}}sqrt_f32:
; SI-CHECK: v_sqrt_f32_e32
define void @sqrt_f32(float addrspace(1)* %out, float %in) {
entry:
  %0 = call float @llvm.sqrt.f32(float %in)
  store float %0, float addrspace(1)* %out
  ret void
}

; R600-CHECK-LABEL: {{^}}sqrt_v2f32:
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[2].W
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[2].W, PS
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].X
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].X, PS
; SI-CHECK-LABEL: {{^}}sqrt_v2f32:
; SI-CHECK: v_sqrt_f32_e32
; SI-CHECK: v_sqrt_f32_e32
define void @sqrt_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; R600-CHECK-LABEL: {{^}}sqrt_v4f32:
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].Y
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].Y, PS
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].Z
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].Z, PS
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].W
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].W, PS
; R600-CHECK-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[4].X
; R600-CHECK-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[4].X, PS
; SI-CHECK-LABEL: {{^}}sqrt_v4f32:
; SI-CHECK: v_sqrt_f32_e32
; SI-CHECK: v_sqrt_f32_e32
; SI-CHECK: v_sqrt_f32_e32
; SI-CHECK: v_sqrt_f32_e32
define void @sqrt_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %in)
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

declare float @llvm.sqrt.f32(float %in)
declare <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float> %in)
