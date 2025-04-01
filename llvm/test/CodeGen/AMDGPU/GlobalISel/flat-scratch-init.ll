; RUN: opt -passes=amdgpu-attributor -mcpu=gfx900 < %s | llc -mcpu=gfx900 | FileCheck -check-prefixes=GCN,RW-FLAT %s
; RUN: opt -passes=amdgpu-attributor -mcpu=gfx900 -mattr=+architected-flat-scratch < %s | llc | FileCheck -check-prefixes=GCN,RO-FLAT %s
; RUN: opt -passes=amdgpu-attributor -mcpu=gfx942 < %s | llc | FileCheck -check-prefixes=GCN,RO-FLAT %s

target triple = "amdgcn-amd-amdhsa"

; Make sure flat_scratch_init is set

; GCN-LABEL: {{^}}stack_object_addrspacecast_in_kernel_no_calls:
; RW-FLAT:     s_add_u32 flat_scratch_lo, s4, s7
; RW-FLAT:     s_addc_u32 flat_scratch_hi, s5, 0
; RO-FLAT-NOT: flat_scratch
; GCN:         flat_store_dword
; RO-FLAT-NOT: .amdhsa_user_sgpr_private_segment_buffer
; RW-FLAT:     .amdhsa_user_sgpr_flat_scratch_init 1
; RO-FLAT-NOT: .amdhsa_user_sgpr_flat_scratch_init
; RW-FLAT:     .amdhsa_system_sgpr_private_segment_wavefront_offset
; RW-FLAT-NOT: .amdhsa_enable_private_segment
; RO-FLAT-NOT: .amdhsa_system_sgpr_private_segment_wavefront_offset
; RO-FLAT:     .amdhsa_enable_private_segment 1
; GCN:         COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; RW-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 6
; RO-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 0
define amdgpu_kernel void @stack_object_addrspacecast_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  %cast = addrspacecast ptr addrspace(5) %alloca to ptr
  store volatile i32 0, ptr %cast
  ret void
}

; TODO: Could optimize out in this case
; GCN-LABEL: {{^}}stack_object_in_kernel_no_calls:
; RO-FLAT-NOT: flat_scratch
; RW-FLAT:     buffer_store_dword
; RO-FLAT:     scratch_store_dword
; RW-FLAT:     .amdhsa_user_sgpr_private_segment_buffer 1
; RO-FLAT-NOT: .amdhsa_user_sgpr_private_segment_buffer
; RW-FLAT:     .amdhsa_user_sgpr_flat_scratch_init 1
; RO-FLAT-NOT: .amdhsa_user_sgpr_flat_scratch_init
; RW-FLAT:     .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; RW-FLAT-NOT: .amdhsa_enable_private_segment
; RO-FLAT-NOT: .amdhsa_system_sgpr_private_segment_wavefront_offset
; RO-FLAT:     .amdhsa_enable_private_segment 1
; RW-FLAT:     .amdhsa_reserve_flat_scratch 0
; GCN:         COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; RW-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 6
; RO-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 0
define amdgpu_kernel void @stack_object_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca
  ret void
}

; GCN-LABEL: {{^}}kernel_no_calls_no_stack:
; GCN-NOT:    flat_scratch
; RW-FLAT:     .amdhsa_user_sgpr_private_segment_buffer 1
; RO-FLAT-NOT: .amdhsa_user_sgpr_private_segment_buffer
; RW-FLAT:     .amdhsa_user_sgpr_flat_scratch_init 0
; RO-FLAT-NOT: .amdhsa_user_sgpr_flat_scratch_init
; RW-FLAT:     .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; RW-FLAT-NOT: .amdhsa_enable_private_segment
; RO-FLAT-NOT: .amdhsa_system_sgpr_private_segment_wavefront_offset
; RO-FLAT:     .amdhsa_enable_private_segment 0
; RW-FLAT:     .amdhsa_reserve_flat_scratch 0
; GCN:         COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; RW-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 4
; RO-FLAT:     COMPUTE_PGM_RSRC2:USER_SGPR: 0
define amdgpu_kernel void @kernel_no_calls_no_stack() {
  ret void
}
