; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rcp_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv float 1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fast_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_fast_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv fast float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_arcp_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_arcp_ulp25_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv arcp float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_global_fast_ulp25_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e32 [[RCP:v[0-9]+]], [[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_global_fast_ulp25_pat_f32(float addrspace(1)* %out, float %src) #2 {
  %rcp = fdiv float 1.0, %src, !fpmath !0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fabs_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], |[[SRC]]|
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @rcp_fabs_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %src.fabs = call float @llvm.fabs.f32(float %src)
  %rcp = fdiv float 1.0, %src.fabs
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}neg_rcp_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], -[[SRC]]
; GCN: buffer_store_dword [[RCP]]

; EG: RECIP_IEEE
define amdgpu_kernel void @neg_rcp_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %rcp = fdiv float -1.0, %src
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fabs_fneg_pat_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], -|[[SRC]]|
; GCN: buffer_store_dword [[RCP]]
define amdgpu_kernel void @rcp_fabs_fneg_pat_f32(float addrspace(1)* %out, float %src) #0 {
  %src.fabs = call float @llvm.fabs.f32(float %src)
  %src.fabs.fneg = fsub float -0.0, %src.fabs
  %rcp = fdiv float 1.0, %src.fabs.fneg
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}rcp_fabs_fneg_pat_multi_use_f32:
; GCN: s_load_dword [[SRC:s[0-9]+]]
; GCN: v_rcp_f32_e64 [[RCP:v[0-9]+]], -|[[SRC]]|
; GCN: v_mul_f32_e64 [[MUL:v[0-9]+]], [[SRC]], -|[[SRC]]|
; GCN: buffer_store_dword [[RCP]]
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @rcp_fabs_fneg_pat_multi_use_f32(float addrspace(1)* %out, float %src) #0 {
  %src.fabs = call float @llvm.fabs.f32(float %src)
  %src.fabs.fneg = fsub float -0.0, %src.fabs
  %rcp = fdiv float 1.0, %src.fabs.fneg
  store volatile float %rcp, float addrspace(1)* %out, align 4

  %other = fmul float %src, %src.fabs.fneg
  store volatile float %other, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_2_x_pat_f32:
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 0.5, v{{[0-9]+}}
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @div_arcp_2_x_pat_f32(float addrspace(1)* %out) #0 {
  %x = load float, float addrspace(1)* undef
  %rcp = fdiv arcp float %x, 2.0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_k_x_pat_f32:
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 0x3dcccccd, v{{[0-9]+}}
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @div_arcp_k_x_pat_f32(float addrspace(1)* %out) #0 {
  %x = load float, float addrspace(1)* undef
  %rcp = fdiv arcp float %x, 10.0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}div_arcp_neg_k_x_pat_f32:
; GCN: v_mul_f32_e32 [[MUL:v[0-9]+]], 0xbdcccccd, v{{[0-9]+}}
; GCN: buffer_store_dword [[MUL]]
define amdgpu_kernel void @div_arcp_neg_k_x_pat_f32(float addrspace(1)* %out) #0 {
  %x = load float, float addrspace(1)* undef
  %rcp = fdiv arcp float %x, -10.0
  store float %rcp, float addrspace(1)* %out, align 4
  ret void
}

declare float @llvm.fabs.f32(float) #1
declare float @llvm.sqrt.f32(float) #1

attributes #0 = { nounwind "unsafe-fp-math"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind "unsafe-fp-math"="true" }

!0 = !{float 2.500000e+00}
