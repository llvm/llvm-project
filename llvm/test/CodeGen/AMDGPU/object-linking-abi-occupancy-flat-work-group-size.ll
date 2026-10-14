; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W64 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W32 %s
; RUN: llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W64 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W32 %s
; RUN: llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W64 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W32 %s
; RUN: llc -mtriple=amdgpu12.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking < %t/lower.ll | FileCheck --check-prefixes=LOWER,LOWER-W64 %s
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX9 %s
; RUN: not llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX10 %s
; RUN: not llc -mtriple=amdgpu10.10-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX10 %s
; RUN: not llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX11 %s
; RUN: not llc -mtriple=amdgpu11.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX11 %s
; RUN: not llc -mtriple=amdgpu12.00-amd-amdhsa -mattr=+wavefrontsize32 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX12 %s
; RUN: not llc -mtriple=amdgpu12.00-amd-amdhsa -mattr=+wavefrontsize64 -amdgpu-enable-object-linking -filetype=null < %t/wins.ll 2>&1 | FileCheck --check-prefix=GFX12 %s

; amdgpu-flat-work-group-size is ABI-significant, so it defines the ABI
; occupancy on its own.

; A flat workgroup size of 512 needs fewer waves than the default 1024, so it
; lowers the ABI occupancy.
; LOWER-LABEL: {{^}}fixed_vgpr_flat_lower:
; LOWER: .set .Lfixed_vgpr_flat_lower.num_vgpr, 71
; LOWER-W64: .amdgpu_occupancy 2
; LOWER-W32: .amdgpu_occupancy 4

; The attribute also takes precedence over the amdgpu_abi_waves_per_eu module
; flag. Even though the flag asks for 2 waves/EU, a flat workgroup size of 1024
; requires the per-target occupancy that a 1024-workitem workgroup implies.
; GFX9: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (64) in function 'fixed_vgpr_flat'
; GFX10: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (128) in function 'fixed_vgpr_flat'
; GFX11: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (192) in function 'fixed_vgpr_flat'
; GFX12: error: {{.*}}VGPRs under object-linking ABI (193) exceeds limit (192) in function 'fixed_vgpr_flat'

;--- lower.ll
define amdgpu_kernel void @fixed_vgpr_flat_lower(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,512" }

;--- wins.ll
define amdgpu_kernel void @fixed_vgpr_flat(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v192}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,1024" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}
