; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking < %s | FileCheck -check-prefix=ABI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=NOABI %s

; Kernel with no amdgpu-waves-per-eu attribute. Under object linking the ABI
; occupancy is used as a register-budget floor, but it is not modeled as an
; exact maximum occupancy request. The kernel descriptor therefore still reports
; the local resource usage and resulting occupancy. Per-function resource
; reporting is local-only under object linking, so no module-level
; `amdgpu.max_num_*` symbols are emitted.

; ABI-LABEL: {{^}}kernel_no_attr:
; ABI: .set .Lkernel_no_attr.num_vgpr, 1
; ABI: NumVGPRsForWavesPerEU: 1
; ABI: Occupancy: 10
; ABI: .amdgpu_occupancy 4

; ABI-NOT: amdgpu.max_num_

; NOABI-LABEL: {{^}}kernel_no_attr:
; NOABI: .set .Lkernel_no_attr.num_vgpr, 1
; NOABI: NumVGPRsForWavesPerEU: 1
; NOABI: Occupancy: 10

; NOABI: .set amdgpu.max_num_vgpr, 0
; NOABI: .set amdgpu.max_num_agpr, 0
; NOABI: .set amdgpu.max_num_sgpr, 0

define amdgpu_kernel void @kernel_no_attr(ptr addrspace(1) %p) {
  store i32 0, ptr addrspace(1) %p
  ret void
}
