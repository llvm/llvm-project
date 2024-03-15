; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,WORKAROUND %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,WORKAROUND %s

; Does not apply to wave64
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize64 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -mattr=+wavefrontsize64 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s

; Does not apply to gfx1101
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1101 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1101 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s

; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1102 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,WORKAROUND %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1102 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,WORKAROUND %s

; Does not apply to gfx1103
; RUN: llc -global-isel=0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1103 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s
; RUN: llc -global-isel=1 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1103 -amdgpu-enable-vopd=0 < %s | FileCheck -check-prefixes=GCN,NOWORKAROUND %s

; There aren't any stack objects, but we still enable the
; private_segment_wavefront_offset to get to 16, and the workgroup ID
; is in s14.

; private_segment_buffer + workgroup_id_x = 5, + 11 padding

; GCN-LABEL: {{^}}minimal_kernel_inputs:
; WORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s15
; NOWORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s0
; GCN-NEXT: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V]], off

; GCN: .amdhsa_kernel minimal_kernel_inputs
; WORKAROUND: .amdhsa_user_sgpr_count 15
; NOWORKAROUND: .amdhsa_user_sgpr_count 0
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; GCN-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; GCN-NEXT: .amdhsa_wavefront_size32
; GCN-NEXT: .amdhsa_uses_dynamic_stack 0
; GCN-NEXT: .amdhsa_enable_private_segment 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; GCN-NEXT: .amdhsa_system_vgpr_workitem_id 0
; WORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; NOWORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 0
define amdgpu_kernel void @minimal_kernel_inputs() {
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}minimal_kernel_inputs_with_stack:
; WORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s15
; NOWORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s0
; GCN: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V]], off

; GCN: .amdhsa_kernel minimal_kernel_inputs
; WORKAROUND: .amdhsa_user_sgpr_count 15
; NOWORKAROUND: .amdhsa_user_sgpr_count 0
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; GCN-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; GCN-NEXT: .amdhsa_wavefront_size32
; GCN-NEXT: .amdhsa_uses_dynamic_stack 0
; GCN-NEXT: .amdhsa_enable_private_segment 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; GCN-NEXT: .amdhsa_system_vgpr_workitem_id 0
; WORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; NOWORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 0
define amdgpu_kernel void @minimal_kernel_inputs_with_stack() {
  %alloca = alloca i32, addrspace(5)
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id, ptr addrspace(1) undef
  store volatile i32 0, ptr addrspace(5) %alloca
  ret void
}

; GCN-LABEL: {{^}}queue_ptr:
; GCN: global_load_u8 v{{[0-9]+}},

; WORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s15
; NOWORKAROUND: v_mov_b32_e32 [[V:v[0-9]+]], s2
; GCN-NEXT: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V]], off

; GCN: .amdhsa_kernel queue_ptr
; WORKAROUND: .amdhsa_user_sgpr_count 15
; NOWORKAROUND: .amdhsa_user_sgpr_count 2
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_id 0
; GCN-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; GCN-NEXT: .amdhsa_wavefront_size32
; GCN-NEXT: .amdhsa_uses_dynamic_stack 0
; GCN-NEXT: .amdhsa_enable_private_segment 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_y 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_z 0
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; GCN-NEXT: .amdhsa_system_vgpr_workitem_id 0
; WORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 15
; NOWORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 2
define amdgpu_kernel void @queue_ptr() {
  %queue.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %load = load volatile i8, ptr addrspace(4) %queue.ptr
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}all_inputs:
; WORKAROUND: v_mov_b32_e32 [[V_X:v[0-9]+]], s13
; WORKAROUND: v_mov_b32_e32 [[V_Y:v[0-9]+]], s14
; WORKAROUND: v_mov_b32_e32 [[V_Z:v[0-9]+]], s15

; NOWORKAROUND: v_mov_b32_e32 [[V_X:v[0-9]+]], s6
; NOWORKAROUND: v_mov_b32_e32 [[V_Y:v[0-9]+]], s7
; NOWORKAROUND: v_mov_b32_e32 [[V_Z:v[0-9]+]], s8

; GCN: global_load_u8 v{{[0-9]+}}, v{{[0-9]+}}, s[0:1]
; GCN: global_load_u8 v{{[0-9]+}},
; GCN: global_load_u8 v{{[0-9]+}}, v{{[0-9]+}}, s[2:3]

; GCN-DAG: v_mov_b32_e32 v[[DISPATCH_LO:[0-9]+]], s4
; GCN-DAG: v_mov_b32_e32 v[[DISPATCH_HI:[0-9]+]], s5

; GCN: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V_X]], off
; GCN: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V_Y]], off
; GCN: global_store_b32 v{{\[[0-9]+:[0-9]+\]}}, [[V_Z]], off
; GCN: global_store_b64 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[DISPATCH_LO]]:[[DISPATCH_HI]]{{\]}}, off

; GCN: .amdhsa_kernel all_inputs
; WORKAROUND: .amdhsa_user_sgpr_count 13
; NOWORKAROUND: .amdhsa_user_sgpr_count 6
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_ptr 1
; GCN-NEXT: .amdhsa_user_sgpr_queue_ptr 0
; GCN-NEXT: .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GCN-NEXT: .amdhsa_user_sgpr_dispatch_id 1
; GCN-NEXT: .amdhsa_user_sgpr_private_segment_size 0
; GCN-NEXT: .amdhsa_wavefront_size32
; GCN-NEXT: .amdhsa_uses_dynamic_stack 0
; GCN-NEXT: .amdhsa_enable_private_segment 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_x 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_y 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_id_z 1
; GCN-NEXT: .amdhsa_system_sgpr_workgroup_info 0
; GCN-NEXT: .amdhsa_system_vgpr_workitem_id 0
; WORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 13
; NOWORKAROUND: ; COMPUTE_PGM_RSRC2:USER_SGPR: 6
define amdgpu_kernel void @all_inputs() {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, ptr addrspace(5) %alloca

  %dispatch.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %load.dispatch = load volatile i8, ptr addrspace(4) %dispatch.ptr

  %queue.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr()
  %load.queue = load volatile i8, ptr addrspace(4) %queue.ptr

  %implicitarg.ptr = call noalias ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr()
  %load.implicitarg = load volatile i8, ptr addrspace(4) %implicitarg.ptr

  %id.x = call i32 @llvm.amdgcn.workgroup.id.x()
  store volatile i32 %id.x, ptr addrspace(1) undef

  %id.y = call i32 @llvm.amdgcn.workgroup.id.y()
  store volatile i32 %id.y, ptr addrspace(1) undef

  %id.z = call i32 @llvm.amdgcn.workgroup.id.z()
  store volatile i32 %id.z, ptr addrspace(1) undef

  %dispatch.id = call i64 @llvm.amdgcn.dispatch.id()
  store volatile i64 %dispatch.id, ptr addrspace(1) undef

  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0
declare align 4 ptr addrspace(4) @llvm.amdgcn.implicitarg.ptr() #0
declare align 4 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
declare align 4 ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
declare align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #0
declare i64 @llvm.amdgcn.dispatch.id() #0

attributes #0 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
