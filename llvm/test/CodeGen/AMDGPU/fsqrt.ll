; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s


; Run with unsafe-fp-math to make sure nothing tries to turn this into 1 / rsqrt(x)

; FUNC-LABEL: {{^}}v_safe_fsqrt_f32:
; GCN: v_sqrt_f32_e32 {{v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @v_safe_fsqrt_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #1 {
  %r0 = load float, ptr addrspace(1) %in
  %r1 = call float @llvm.sqrt.f32(float %r0)
  store float %r1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}v_unsafe_fsqrt_f32:
; GCN: v_sqrt_f32_e32 {{v[0-9]+, v[0-9]+}}
define amdgpu_kernel void @v_unsafe_fsqrt_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #2 {
  %r0 = load float, ptr addrspace(1) %in
  %r1 = call float @llvm.sqrt.f32(float %r0)
  store float %r1, ptr addrspace(1) %out
  ret void
}


; FUNC-LABEL: {{^}}s_sqrt_f32:
; GCN: v_sqrt_f32_e32

; R600: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[2].Z
; R600: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, PS
define amdgpu_kernel void @s_sqrt_f32(ptr addrspace(1) %out, float %in) #1 {
entry:
  %fdiv = call float @llvm.sqrt.f32(float %in)
  store float %fdiv, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_sqrt_v2f32:
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32

; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[2].W
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, PS
; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[3].X
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}}
define amdgpu_kernel void @s_sqrt_v2f32(ptr addrspace(1) %out, <2 x float> %in) #1 {
entry:
  %fdiv = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  store <2 x float> %fdiv, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}s_sqrt_v4f32:
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32

; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[3].Y
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, PS
; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[3].Z
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}}
; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[3].W
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}}
; R600-DAG: RECIPSQRT_IEEE * T{{[0-9]\.[XYZW]}}, KC0[4].X
; R600-DAG: RECIP_IEEE * T{{[0-9]\.[XYZW]}}, T{{[0-9]\.[XYZW]}}
define amdgpu_kernel void @s_sqrt_v4f32(ptr addrspace(1) %out, <4 x float> %in) #1 {
entry:
  %fdiv = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %in)
  store <4 x float> %fdiv, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}elim_redun_check_neg0:
; GCN: v_sqrt_f32_e32
; GCN-NOT: v_cndmask
define amdgpu_kernel void @elim_redun_check_neg0(ptr addrspace(1) %out, float %in) #1 {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp olt float %in, -0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}elim_redun_check_pos0:
; GCN: v_sqrt_f32_e32
; GCN-NOT: v_cndmask
define amdgpu_kernel void @elim_redun_check_pos0(ptr addrspace(1) %out, float %in) #1 {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp olt float %in, 0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}elim_redun_check_ult:
; GCN: v_sqrt_f32_e32
; GCN-NOT: v_cndmask
define amdgpu_kernel void @elim_redun_check_ult(ptr addrspace(1) %out, float %in) #1 {
entry:
  %sqrt = call float @llvm.sqrt.f32(float %in)
  %cmp = fcmp ult float %in, -0.000000e+00
  %res = select i1 %cmp, float 0x7FF8000000000000, float %sqrt
  store float %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}elim_redun_check_v2:
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32
; GCN-NOT: v_cndmask
define amdgpu_kernel void @elim_redun_check_v2(ptr addrspace(1) %out, <2 x float> %in) #1 {
entry:
  %sqrt = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  %cmp = fcmp olt <2 x float> %in, <float -0.000000e+00, float -0.000000e+00>
  %res = select <2 x i1> %cmp, <2 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000>, <2 x float> %sqrt
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}elim_redun_check_v2_ult
; GCN: v_sqrt_f32_e32
; GCN: v_sqrt_f32_e32
; GCN-NOT: v_cndmask
define amdgpu_kernel void @elim_redun_check_v2_ult(ptr addrspace(1) %out, <2 x float> %in) #1 {
entry:
  %sqrt = call <2 x float> @llvm.sqrt.v2f32(<2 x float> %in)
  %cmp = fcmp ult <2 x float> %in, <float -0.000000e+00, float -0.000000e+00>
  %res = select <2 x i1> %cmp, <2 x float> <float 0x7FF8000000000000, float 0x7FF8000000000000>, <2 x float> %sqrt
  store <2 x float> %res, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}recip_sqrt:
; R600: RECIPSQRT_IEEE
; R600-NOT: RECIP_IEEE
define amdgpu_kernel void @recip_sqrt(ptr addrspace(1) %out, float %src) nounwind {
  %sqrt = call float @llvm.sqrt.f32(float %src)
  %recipsqrt = fdiv fast float 1.0, %sqrt
  store float %recipsqrt, ptr addrspace(1) %out, align 4
  ret void
}

declare float @llvm.sqrt.f32(float %in) #0
declare <2 x float> @llvm.sqrt.v2f32(<2 x float> %in) #0
declare <4 x float> @llvm.sqrt.v4f32(<4 x float> %in) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "unsafe-fp-math"="false" }
attributes #2 = { nounwind "unsafe-fp-math"="true" }
