; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W64 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W32 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %s | FileCheck --check-prefixes=GCN,W64 %s

; amdgpu-flat-work-group-size is ABI-significant and can lower the ABI
; occupancy implied by the default 1024-workitem workgroup.

; GCN-LABEL: {{^}}fixed_vgpr_flat_lower:
; GCN: .set .Lfixed_vgpr_flat_lower.num_vgpr, 71
; W64: .amdgpu_occupancy 2
; W32: .amdgpu_occupancy 4
; GCN-NOT: amdgpu.max_num_

define amdgpu_kernel void @fixed_vgpr_flat_lower(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,512" }
