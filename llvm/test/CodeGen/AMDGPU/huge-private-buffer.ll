; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH256K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=-wavefrontsize32,+wavefrontsize64 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH256K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=-wavefrontsize32,+wavefrontsize64 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH1024K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH2048K %s

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo16:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0xfffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo16() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 65535
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo17:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; SCRATCH128K-NOT: v_and_b32
; SCRATCH256K: v_and_b32_e32 v{{[0-9]+}}, 0x1fffc, [[FI]]
; SCRATCH1024K: v_and_b32_e32 v{{[0-9]+}}, 0x1fffc, [[FI]]
; SCRATCH2048K: v_and_b32_e32 v{{[0-9]+}}, 0x1fffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo17() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 131071
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo18:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; SCRATCH128K-NOT: v_and_b32
; SCRATCH256K-NOT: v_and_b32
; SCRATCH1024K: v_and_b32_e32 v{{[0-9]+}}, 0x3fffc, [[FI]]
; SCRATCH2048K: v_and_b32_e32 v{{[0-9]+}}, 0x3fffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo18() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 262143
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo20:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; SCRATCH128K-NOT: v_and_b32
; SCRATCH256K-NOT: v_and_b32
; SCRATCH1024K-NOT: v_and_b32
; SCRATCH2048K: v_and_b32_e32 v{{[0-9]+}}, 0xffffc, [[FI]]
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo20() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 1048575
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo21:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 4
; GCN-NOT: v_and_b32
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo21() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 2097151
  store volatile i32 %masked, ptr addrspace(1) undef
  ret void
}
