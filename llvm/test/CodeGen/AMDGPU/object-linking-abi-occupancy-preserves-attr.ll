; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking < %s | FileCheck -check-prefix=ABI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=NOABI %s

; Object-linking ABI occupancy is a budget floor, not a synthetic
; amdgpu-waves-per-eu range. A max-occupancy hint that is weaker than the ABI
; floor does not force resource inflation below the ABI floor.
;
; Here the kernel requests amdgpu-waves-per-eu="1,1", but the effective budget
; remains compatible with the ABI. The kernel uses 30 VGPRs via inline asm, so
; the final occupancy is resource-derived. No module-level `amdgpu.max_num_*`
; symbols are emitted under object linking.

; ABI-LABEL: {{^}}kernel_with_attr:
; ABI: .set .Lkernel_with_attr.num_vgpr, 30
; ABI: NumVGPRsForWavesPerEU: 30
; ABI: Occupancy: 8
; ABI: .amdgpu_occupancy 4

; ABI-NOT: amdgpu.max_num_

; NOABI-LABEL: {{^}}kernel_with_attr:
; NOABI: .set .Lkernel_with_attr.num_vgpr, 30
; NOABI: NumVGPRsForWavesPerEU: 30
; NOABI: Occupancy: 8

; NOABI: .set amdgpu.max_num_vgpr, 0

define amdgpu_kernel void @kernel_with_attr(ptr addrspace(1) %p) #0 {
  %r = call { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 } asm sideeffect "; clobber", "=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v,=v"()
  store i32 0, ptr addrspace(1) %p
  ret void
}

attributes #0 = { "amdgpu-waves-per-eu"="1,1" }
