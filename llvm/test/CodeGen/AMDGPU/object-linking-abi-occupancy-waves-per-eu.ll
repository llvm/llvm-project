; RUN: split-file %s %t
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking < %t/below.ll | FileCheck -check-prefix=BELOW %s
; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking -filetype=null < %t/above.ll 2>&1 | FileCheck -check-prefix=ABOVE %s

; amdgpu-waves-per-eu is a hint and never defines the ABI occupancy. The ABI
; occupancy replaces the workgroup-derived minimum that the hint is validated
; against, so a hint below the ABI occupancy is rejected and leaves the budget
; alone, while a hint above it is accepted and tightens the budget.
;
; Both modules override the ABI occupancy to 2 waves/EU, which gives a
; 128 VGPR budget on gfx900. Both kernels use 71 VGPRs.

; The "1,1" hint asks for fewer waves than the ABI occupancy, so it is dropped
; and the 128 VGPR budget still applies. The rejected hint is not turned into an
; exact occupancy request either, so the kernel descriptor keeps the
; resource-derived occupancy.
; BELOW-LABEL: {{^}}hint_below_abi:
; BELOW: .set .Lhint_below_abi.num_vgpr, 71
; BELOW: NumVGPRsForWavesPerEU: 71
; BELOW: Occupancy: 3
; BELOW: .amdgpu_occupancy 2

; The "4,4" hint asks for more waves than the ABI occupancy, so it is accepted
; and lowers the budget to 64 VGPRs.
; ABOVE: error: {{.*}}VGPRs under object-linking ABI (71) exceeds limit (64) in function 'hint_above_abi'

;--- below.ll
define amdgpu_kernel void @hint_below_abi(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}

;--- above.ll
define amdgpu_kernel void @hint_above_abi(ptr addrspace(1) %p) #0 {
  call void asm sideeffect "; clobber", "~{v70}"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="4,4" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_abi_waves_per_eu", i32 2}
