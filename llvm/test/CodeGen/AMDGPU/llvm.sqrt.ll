; RUN: llc < %s -march=r600 --mcpu=redwood | FileCheck %s --check-prefix=R600
; RUN: llc < %s -march=amdgcn --mcpu=SI -verify-machineinstrs| FileCheck %s --check-prefix=SI
; RUN: llc < %s -march=amdgcn --mcpu=tonga -verify-machineinstrs| FileCheck %s --check-prefix=SI

; R600-LABEL: {{^}}sqrt_f32:
; R600: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[2].Z
; R600: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[2].Z, PS
; SI-LABEL: {{^}}sqrt_f32:
; SI: v_sqrt_f32_e32
define void @sqrt_f32(float addrspace(1)* %out, float %in) {
entry:
  %0 = call float @llvm.sqrt.f32(float %in)
  store float %0, float addrspace(1)* %out
  ret void
}

; R600-LABEL: {{^}}sqrt_v2f32:
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[2].W
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[2].W, PS
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].X
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].X, PS
; SI-LABEL: {{^}}sqrt_v2f32:
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
define void @sqrt_v2f32(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %0 = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  store <2 x float> %0, <2 x float> addrspace(1)* %out
  ret void
}

; R600-LABEL: {{^}}sqrt_v4f32:
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].Y
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].Y, PS
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].Z
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].Z, PS
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[3].W
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[3].W, PS
; R600-DAG: RECIPSQRT_CLAMPED * T{{[0-9]\.[XYZW]}}, KC0[4].X
; R600-DAG: MUL NON-IEEE T{{[0-9]\.[XYZW]}}, KC0[4].X, PS
; SI-LABEL: {{^}}sqrt_v4f32:
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
define void @sqrt_v4f32(<4 x float> addrspace(1)* %out, <4 x float> %in) {
entry:
  %0 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %in)
  store <4 x float> %0, <4 x float> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}elim_redun_check_neg0:
; SI: v_sqrt_f32_e32
; SI-NOT: v_cndmask
define void @elim_redun_check_neg0(float addrspace(1)* %out, float %in) {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp olt float %in, -0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, float addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}elim_redun_check_pos0:
; SI: v_sqrt_f32_e32
; SI-NOT: v_cndmask
define void @elim_redun_check_pos0(float addrspace(1)* %out, float %in) {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp olt float %in, 0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, float addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}elim_redun_check_ult:
; SI: v_sqrt_f32_e32
; SI-NOT: v_cndmask
define void @elim_redun_check_ult(float addrspace(1)* %out, float %in) {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp ult float %in, -0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, float addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}elim_redun_check_v2:
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
; SI-NOT: v_cndmask
define void @elim_redun_check_v2(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %sqrt = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  %cmp = fcmp olt <2 x float> %in, <float -0.000000e+00, float -0.000000e+00>
  %res = select <2 x i1> %cmp, <2 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000>, <2 x float> %sqrt
  store <2 x float> %res, <2 x float> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}elim_redun_check_v2_ult
; SI: v_sqrt_f32_e32
; SI: v_sqrt_f32_e32
; SI-NOT: v_cndmask
define void @elim_redun_check_v2_ult(<2 x float> addrspace(1)* %out, <2 x float> %in) {
entry:
  %sqrt = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  %cmp = fcmp ult <2 x float> %in, <float -0.000000e+00, float -0.000000e+00>
  %res = select <2 x i1> %cmp, <2 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000>, <2 x float> %sqrt
  store <2 x float> %res, <2 x float> addrspace(1)* %out
  ret void
}

declare float @llvm.sqrt.f32(float %in)
declare <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
declare <4 x float> @llvm.sqrt.v4f32(<4 x float> %in)
