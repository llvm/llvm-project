; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-enable-object-linking < %s | FileCheck -check-prefix=ABI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=NOABI %s

; Device function with no amdgpu-waves-per-eu attribute. Under object linking
; the ABI occupancy is used as a register-budget floor without changing the
; source-level waves-per-EU range. Per-function resource reporting is local-only
; under object linking, so only the function-level `.num_vgpr` / `.num_agpr`
; symbols are emitted -- the module-level `amdgpu.max_num_*` symbols are
; suppressed.

; ABI-LABEL: {{^}}device_fn:
; ABI: .set .Ldevice_fn.num_vgpr, 3
; ABI: .set .Ldevice_fn.num_agpr, 0

; ABI-NOT: amdgpu.max_num_

; NOABI-LABEL: {{^}}device_fn:
; NOABI: .set .Ldevice_fn.num_vgpr, 3
; NOABI: .set .Ldevice_fn.num_agpr, 0

; NOABI: .set amdgpu.max_num_vgpr, 3
; NOABI: .set amdgpu.max_num_agpr, 0

define void @device_fn(ptr addrspace(1) %p) {
  store i32 0, ptr addrspace(1) %p
  ret void
}
