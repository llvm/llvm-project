; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -check-prefix=R600 -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}fmul_f32:
; GCN: v_mul_f32

; R600: MUL_IEEE {{\** *}}{{T[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].W
define amdgpu_kernel void @fmul_f32(ptr addrspace(1) %out, float %a, float %b) {
entry:
  %0 = fmul float %a, %b
  store float %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fmul_v2f32:
; GCN: v_mul_f32
; GCN: v_mul_f32

; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW]}}
define amdgpu_kernel void @fmul_v2f32(ptr addrspace(1) %out, <2 x float> %a, <2 x float> %b) {
entry:
  %0 = fmul <2 x float> %a, %b
  store <2 x float> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}fmul_v4f32:
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32

; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; R600: MUL_IEEE {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
define amdgpu_kernel void @fmul_v4f32(ptr addrspace(1) %out, ptr addrspace(1) %in) {
  %b_ptr = getelementptr <4 x float>, ptr addrspace(1) %in, i32 1
  %a = load <4 x float>, ptr addrspace(1) %in
  %b = load <4 x float>, ptr addrspace(1) %b_ptr
  %result = fmul <4 x float> %a, %b
  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_mul_2_k:
; GCN: v_mul_f32
; GCN-NOT: v_mul_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_mul_2_k(ptr addrspace(1) %out, float %x) #0 {
  %y = fmul float %x, 2.0
  %z = fmul float %y, 3.0
  store float %z, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_mul_2_k_inv:
; GCN: v_mul_f32
; GCN-NOT: v_mul_f32
; GCN-NOT: v_mad_f32
; GCN: s_endpgm
define amdgpu_kernel void @test_mul_2_k_inv(ptr addrspace(1) %out, float %x) #0 {
  %y = fmul float %x, 3.0
  %z = fmul float %y, 2.0
  store float %z, ptr addrspace(1) %out
  ret void
}

; There should be three multiplies here; %a should be used twice (once
; negated), not duplicated into mul x, 5.0 and mul x, -5.0.
; FUNC-LABEL: {{^}}test_mul_twouse:
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN: v_mul_f32
; GCN-NOT: v_mul_f32
define amdgpu_kernel void @test_mul_twouse(ptr addrspace(1) %out, float %x, float %y) #0 {
  %a = fmul float %x, 5.0
  %b = fsub float -0.0, %a
  %c = fmul float %b, %y
  %d = fmul float %c, %a
  store float %d, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
