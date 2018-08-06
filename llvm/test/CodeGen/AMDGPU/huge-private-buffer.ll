; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; GCN-LABEL: {{^}}scratch_buffer_known_high_bit_small:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN-NOT: [[FI]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[FI]]
define amdgpu_kernel void @scratch_buffer_known_high_bit_small() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %toint = ptrtoint i32 addrspace(5)* %alloca to i32
  %masked = and i32 %toint, 2147483647
  store volatile i32 %masked, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_bit_huge:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN-DAG: buffer_store_dword
; GCN-DAG: v_and_b32_e32 [[MASKED:v[0-9]+]], 0x7ffffffc, [[FI]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]
define amdgpu_kernel void @scratch_buffer_known_high_bit_huge() #1 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %toint = ptrtoint i32 addrspace(5)* %alloca to i32
  %masked = and i32 %toint, 2147483647
  store volatile i32 %masked, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "target-features"="+huge-private-buffer" }
