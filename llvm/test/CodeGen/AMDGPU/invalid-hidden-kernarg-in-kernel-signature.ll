; RUN: not llc -global-isel=1 -global-isel-abort=2 -mtriple=amdgcn--amdhsa -mcpu=gfx942 < %s 2>&1 | FileCheck -check-prefixes=ERROR,GISEL %s
; RUN: not llc -global-isel=0 -mtriple=amdgcn--amdhsa -mcpu=gfx942 < %s 2>&1 | FileCheck -check-prefix=ERROR %s
; RUN: not llc -global-isel=1 -global-isel-abort=2 -amdgpu-ir-lower-kernel-arguments=0 -mtriple=amdgcn--amdhsa -mcpu=gfx942 < %s 2>&1 | FileCheck -check-prefixes=ERROR,GISEL %s
; RUN: not llc -global-isel=0 -amdgpu-ir-lower-kernel-arguments=0 -mtriple=amdgcn--amdhsa -mcpu=gfx942 < %s 2>&1 | FileCheck -check-prefix=ERROR %s

define amdgpu_kernel void @no_free_sgprs_block_count_x_no_preload_diag(ptr addrspace(1) inreg %out, i512 inreg, i32 inreg "amdgpu-hidden-argument" %_hidden_block_count_x) #0 {
; GISEL: warning: Instruction selection used fallback path for no_free_sgprs_block_count_x_no_preload_diag
; ERROR: error: <unknown>:0:0: in function no_free_sgprs_block_count_x_no_preload_diag void (ptr addrspace(1), i512, i32): hidden argument in kernel signature was not preloaded
  store i32 %_hidden_block_count_x, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @preloadremainder_z_no_preload_diag(ptr addrspace(1) inreg %out, i256 inreg, i32 inreg "amdgpu-hidden-argument" %_hidden_block_count_x, i32 inreg "amdgpu-hidden-argument" %_hidden_block_count_y, i32 inreg "amdgpu-hidden-argument" %_hidden_block_count_z, i16 inreg "amdgpu-hidden-argument" %_hidden_group_size_x, i16 inreg "amdgpu-hidden-argument" %_hidden_group_size_y, i16 inreg "amdgpu-hidden-argument" %_hidden_group_size_z, i16 inreg "amdgpu-hidden-argument" %_hidden_remainder_x, i16 inreg "amdgpu-hidden-argument" %_hidden_remainder_y, i16 inreg "amdgpu-hidden-argument" %_hidden_remainder_z) #0 {
; GISEL: warning: Instruction selection used fallback path for preloadremainder_z_no_preload_diag
; ERROR: error: <unknown>:0:0: in function preloadremainder_z_no_preload_diag void (ptr addrspace(1), i256, i32, i32, i32, i16, i16, i16, i16, i16, i16): hidden argument in kernel signature was not preloaded
  %conv = zext i16 %_hidden_remainder_z to i32
  store i32 %conv, ptr addrspace(1) %out
  ret void
}

attributes #0 = { "amdgpu-no-agpr" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "uniform-work-group-size"="false" }
