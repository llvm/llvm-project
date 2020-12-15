; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s

; Make sure flat_scratch_init is set

; GCN-LABEL: {{^}}stack_object_addrspacecast_in_kernel_no_calls:
; GCN:     s_add_u32 flat_scratch_lo, s4, s7
; GCN:     s_addc_u32 flat_scratch_hi, s5, 0
; GCN:     flat_store_dword
; GCN:     .amdhsa_user_sgpr_flat_scratch_init 1
; GCN:     .amdhsa_system_sgpr_private_segment_wavefront_offset
; GCN-NOT: .amdhsa_reserve_flat_scratch
; GCN:     COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; GCN:     COMPUTE_PGM_RSRC2:USER_SGPR: 6
define amdgpu_kernel void @stack_object_addrspacecast_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  %cast = addrspacecast i32 addrspace(5)* %alloca to i32*
  store volatile i32 0, i32* %cast
  ret void
}

; TODO: Could optimize out in this case
; GCN-LABEL: {{^}}stack_object_in_kernel_no_calls:
; GCN:     s_add_u32 flat_scratch_lo, s4, s7
; GCN:     s_addc_u32 flat_scratch_hi, s5, 0
; GCN:     buffer_store_dword
; GCN:     .amdhsa_user_sgpr_private_segment_buffer 1
; GCN:     .amdhsa_user_sgpr_flat_scratch_init 1
; GCN:     .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; GCN-NOT: .amdhsa_reserve_flat_scratch
; GCN:     COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; GCN:     COMPUTE_PGM_RSRC2:USER_SGPR: 6
define amdgpu_kernel void @stack_object_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}kernel_no_calls_no_stack:
; GCN-NOT: flat_scratch
; GCN:     .amdhsa_user_sgpr_private_segment_buffer 1
; GCN:     .amdhsa_user_sgpr_flat_scratch_init 0
; GCN:     .amdhsa_system_sgpr_private_segment_wavefront_offset 0
; GCN:     .amdhsa_reserve_flat_scratch 0
; GCN:     COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; GCN:     COMPUTE_PGM_RSRC2:USER_SGPR: 4
define amdgpu_kernel void @kernel_no_calls_no_stack() {
  ret void
}
