; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}rcp_uint:
; GCN: v_rcp_iflag_f32_e32
define amdgpu_kernel void @rcp_uint(i32 addrspace(1)* %in, float addrspace(1)* %out) {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %cvt = uitofp i32 %load to float
  %div = fdiv float 1.000000e+00, %cvt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}rcp_sint:
; GCN: v_rcp_iflag_f32_e32
define amdgpu_kernel void @rcp_sint(i32 addrspace(1)* %in, float addrspace(1)* %out) {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %cvt = sitofp i32 %load to float
  %div = fdiv float 1.000000e+00, %cvt
  store float %div, float addrspace(1)* %out, align 4
  ret void
}
