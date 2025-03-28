; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize64 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH256K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize64 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH128K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH256K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+wavefrontsize64 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH1024K %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SCRATCH2048K %s

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo16:
; GCN: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; GCN: s_and_b32 s{{[0-9]+}}, [[FI]], 0xfffc
; GCN: v_mov_b32_e32 [[VFI:v[0-9]+]], [[FI]]{{$}}
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}], [[VFI]]
define amdgpu_kernel void @scratch_buffer_known_high_masklo16() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 15, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 65535
  store volatile i32 %masked, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo17:
; SCRATCH256K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH256K: s_and_b32 s{{[0-9]+}}, [[FI]], 0x1fffc

; SCRATCH1024K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH1024K: s_and_b32 s{{[0-9]+}}, [[FI]], 0x1fffc

; SCRATCH2048K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH2048K: s_and_b32 s{{[0-9]+}}, [[FI]], 0x1fffc

; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo17() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 15, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 131071
  store volatile i32 %masked, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo18:
; SCRATCH128K: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; SCRATCH256K: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; SCRATCH128K-NOT: and_b32
; SCRATCH256K-NOT: and_b32

; SCRATCH1024K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH1024K: s_and_b32 s{{[0-9]+}}, [[FI]], 0x3fffc

; SCRATCH2048K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH2048K: s_and_b32 s{{[0-9]+}}, [[FI]], 0x3fffc

; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo18() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 15, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 262143
  store volatile i32 %masked, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo20:
; SCRATCH128K: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; SCRATCH256K: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; SCRATCH1024K: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}

; SCRATCH128K-NOT: and_b32
; SCRATCH256K-NOT: and_b32
; SCRATCH1024K-NOT: and_b32

; SCRATCH2048K: s_mov_b32 [[FI:s[0-9]+]], 0{{$}}
; SCRATCH2048K: s_and_b32 s{{[0-9]+}}, [[FI]], 0xffffc
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo20() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 15, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 1048575
  store volatile i32 %masked, ptr addrspace(1) poison
  ret void
}

; GCN-LABEL: {{^}}scratch_buffer_known_high_masklo21:
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; GCN-NOT: and_b32
; GCN: {{flat|global}}_store_{{dword|b32}} v[{{[0-9]+:[0-9]+}}],
define amdgpu_kernel void @scratch_buffer_known_high_masklo21() {
  %alloca = alloca i32, align 4, addrspace(5)
  store volatile i32 15, ptr addrspace(5) %alloca
  %toint = ptrtoint ptr addrspace(5) %alloca to i32
  %masked = and i32 %toint, 2097151
  store volatile i32 %masked, ptr addrspace(1) poison
  ret void
}
