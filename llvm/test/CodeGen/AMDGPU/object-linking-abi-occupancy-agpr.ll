; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa -amdgpu-enable-object-linking < %t/within.ll | FileCheck -check-prefix=WITHIN %s
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -amdgpu-enable-object-linking -filetype=null < %t/exceeds.ll 2>&1 | FileCheck -check-prefix=EXCEEDS %s

; gfx90a has a unified vector register file, so architected and accumulator
; VGPRs share one ABI register budget. The default ABI occupancy of 4 waves/EU
; gives a 128 register budget, and the check must apply to the combined count
; rather than to each bank on its own.

; 61 architected VGPRs round up to 64, plus 61 AGPRs, is 125 registers.
; WITHIN-LABEL: {{^}}combined_within_budget:
; WITHIN: .set .Lcombined_within_budget.num_vgpr, 61
; WITHIN-NEXT: .set .Lcombined_within_budget.num_agpr, 61

; 121 architected VGPRs round up to 124, plus 121 AGPRs, is 245 registers.
; Neither bank exceeds 128 on its own.
; EXCEEDS: error: {{.*}}VGPRs under object-linking ABI (245) exceeds limit (128) in function 'combined_exceeds_budget'

;--- within.ll
define void @combined_within_budget(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v60},~{a60}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

;--- exceeds.ll
define void @combined_exceeds_budget(ptr addrspace(1) %p) {
  call void asm sideeffect "; clobber", "~{v120},~{a120}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}
