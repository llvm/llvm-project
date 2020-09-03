; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck -check-prefix=GCN %s

; Make sure flat_scratch_init is set

; GCN-LABEL: {{^}}stack_object_addrspacecast_in_kernel_no_calls:
; GCN: .amdhsa_user_sgpr_flat_scratch_init 1
define amdgpu_kernel void @stack_object_addrspacecast_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  %cast = addrspacecast i32 addrspace(5)* %alloca to i32*
  store volatile i32 0, i32* %cast
  ret void
}

; TODO: Could optimize out in this case
; GCN-LABEL: {{^}}stack_object_in_kernel_no_calls:
; GCN: .amdhsa_user_sgpr_flat_scratch_init 1
define amdgpu_kernel void @stack_object_in_kernel_no_calls() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  ret void
}

; GCN-LABEL: {{^}}kernel_no_calls_no_stack:
; GCN: .amdhsa_user_sgpr_flat_scratch_init 0
define amdgpu_kernel void @kernel_no_calls_no_stack() {
  ret void
}
