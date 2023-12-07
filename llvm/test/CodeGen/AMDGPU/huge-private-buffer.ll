; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE64 %s

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo14:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN: v_and_b32_e32 [[MASKED:v[0-9]+]], 0x3ffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]
define amdgpu_kernel void @scratch_buffer_known_high_masklo14() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 16383
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo16:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN: v_and_b32_e32 [[MASKED:v[0-9]+]], 0xfffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]
define amdgpu_kernel void @scratch_buffer_known_high_masklo16() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 65535
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo17:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; WAVE64-NOT: [[FI]]
; WAVE64: {{flat|global}}_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[FI]]

; WAVE32: v_and_b32_e32 [[MASKED:v[0-9]+]], 0x1fffc, [[FI]]
; WAVE32: {{flat|global}}_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]
define amdgpu_kernel void @scratch_buffer_known_high_masklo17() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 131071
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_mask18:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN-NOT: [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[FI]]
define amdgpu_kernel void @scratch_buffer_known_high_mask18() #0 {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 262143
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

attributes #0 = { nounwind }
