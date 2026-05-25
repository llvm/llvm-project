; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa -amdgpu-enable-object-linking < %s | FileCheck -check-prefix=ABI %s
; RUN: llc -mtriple=amdgpu9.00-amd-amdhsa < %s | FileCheck -check-prefix=NOABI %s

; Without an amdgpu-waves-per-eu attribute the object-linking ABI occupancy is
; the occupancy that a 1024-workitem workgroup needs, which is 4 waves/EU on
; gfx900. It applies to kernels and to device functions alike, and it is
; recorded in `.amdgpu.info`.
;
; The ABI occupancy is a register-budget floor, not an exact occupancy request.
; The reported resource usage, the kernel descriptor and the kernel metadata
; therefore stay the same as without object linking.

; ABI: .set .Lkernel_no_attr.num_vgpr, 1
; ABI: NumVGPRsForWavesPerEU: 1
; ABI: Occupancy: 10
; ABI: .set .Ldevice_fn_no_attr.num_vgpr, 3

; ABI: .amdgpu_info kernel_no_attr
; ABI: .amdgpu_num_vgpr 1
; ABI: .amdgpu_occupancy 4
; ABI: .amdgpu_info device_fn_no_attr
; ABI: .amdgpu_num_vgpr 3
; ABI: .amdgpu_occupancy 4

; Per-function resource reporting is local-only under object linking, so the
; module-level `amdgpu.max_num_*` symbols are suppressed.
; ABI-NOT: amdgpu.max_num_

; ABI: .sgpr_count:     14
; ABI: .vgpr_count:     1

; NOABI: .set .Lkernel_no_attr.num_vgpr, 1
; NOABI: NumVGPRsForWavesPerEU: 1
; NOABI: Occupancy: 10
; NOABI: .set .Ldevice_fn_no_attr.num_vgpr, 3
; NOABI: .set amdgpu.max_num_vgpr, 3
; NOABI: .sgpr_count:     14
; NOABI: .vgpr_count:     1

define amdgpu_kernel void @kernel_no_attr(ptr addrspace(1) %p) {
  store i32 0, ptr addrspace(1) %p
  ret void
}

define void @device_fn_no_attr(ptr addrspace(1) %p) {
  store i32 0, ptr addrspace(1) %p
  ret void
}
