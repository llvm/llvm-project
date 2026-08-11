; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking -filetype=null < %t/default.ll 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking < %t/override.ll | FileCheck -check-prefix=OVERRIDE %s

; The default object-linking ABI occupancy on gfx900 is 4 waves/EU, which gives
; a 64 VGPR budget. Overriding the ABI occupancy to 2 waves/EU with the
; amdgpu_abi_waves_per_eu module flag raises the budget to 128 VGPRs.

; DEFAULT: error: {{.*}}VGPRs under object-linking ABI (71) exceeds limit (64) in function 'fixed_vgpr'

; OVERRIDE-LABEL: {{^}}fixed_vgpr:
; OVERRIDE: .set .Lfixed_vgpr.num_vgpr, 71
; OVERRIDE-NOT: amdgpu.max_num_

;--- default.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

;--- override.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}
