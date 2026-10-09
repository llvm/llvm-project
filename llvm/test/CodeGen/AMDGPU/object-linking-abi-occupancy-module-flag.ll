; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking -filetype=null < %t/default.ll 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking -filetype=null < %t/zero.ll 2>&1 | FileCheck -check-prefix=DEFAULT %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking < %t/override.ll | FileCheck -check-prefix=OVERRIDE %s
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking -filetype=null < %t/too-high.ll 2>&1 | FileCheck -check-prefix=HIGH %s

; The amdgpu_abi_waves_per_eu module flag replaces the default ABI occupancy.
; Out-of-range values must not escape the range of occupancies that the
; subtarget supports.
;
; All four modules hold the same kernel. Only the module flag differs.

; The default ABI occupancy on gfx900 is 4 waves/EU, which gives a 64 VGPR
; budget. A flag value of 0 means "no override" and keeps that default, so the
; two modules must report the same limit.
; DEFAULT: error: {{.*}}VGPRs under object-linking ABI (71) exceeds limit (64) in function 'fixed_vgpr'

; An override of 2 waves/EU raises the budget to 128 VGPRs.
; OVERRIDE-LABEL: {{^}}fixed_vgpr:
; OVERRIDE: .set .Lfixed_vgpr.num_vgpr, 71
; OVERRIDE: .amdgpu_occupancy 2

; A value above the maximum occupancy is clamped to it. gfx900 supports at most
; 10 waves/EU, which gives a 24 VGPR budget.
; HIGH: error: {{.*}}VGPRs under object-linking ABI (71) exceeds limit (24) in function 'fixed_vgpr'

;--- default.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

;--- zero.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 0}

;--- override.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}

;--- too-high.ll
define amdgpu_kernel void @fixed_vgpr(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 999}
