; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX9 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX10 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX10 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX11 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX11 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX12 %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1200 -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %s 2>&1 | FileCheck --check-prefix=GFX12 %s

; amdgpu-flat-work-group-size is ABI-significant. Even when the default ABI
; occupancy is overridden to 2 waves/EU by a module flag, a flat workgroup size
; of 1024 requires the per-target occupancy implied by a 1024-workitem
; workgroup.

; GFX9: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (64) in function 'fixed_vgpr_flat'
; GFX10: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (128) in function 'fixed_vgpr_flat'
; GFX11: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (192) in function 'fixed_vgpr_flat'
; GFX12: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (192) in function 'fixed_vgpr_flat'

define amdgpu_kernel void @fixed_vgpr_flat(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v192}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}
