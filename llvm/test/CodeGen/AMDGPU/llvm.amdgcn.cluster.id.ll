; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-attributor %s -o %t.bc
; RUN: llc -mtriple=amdgcn -mcpu=gfx1250 %t.bc -o - | FileCheck --check-prefixes=CHECK-UNKNOWN %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=gfx1250 %t.bc -o - | FileCheck -check-prefixes=CHECK-MESA3D %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=gfx1250 %t.bc -o - | FileCheck --check-prefixes=CHECK-G-UNKNOWN %s
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mcpu=gfx1250 %t.bc -o - | FileCheck -check-prefixes=CHECK-G-MESA3D %s

declare i32 @llvm.amdgcn.cluster.id.x() #0
declare i32 @llvm.amdgcn.cluster.id.y() #0
declare i32 @llvm.amdgcn.cluster.id.z() #0

define amdgpu_kernel void @test_cluster_id_x(ptr addrspace(1) %out) {
; CHECK-UNKNOWN-LABEL: test_cluster_id_x:
; CHECK-UNKNOWN:       ; %bb.0:
; CHECK-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-UNKNOWN-NEXT:    v_dual_mov_b32 v0, ttmp9 :: v_dual_mov_b32 v1, 0
; CHECK-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-UNKNOWN-NEXT:    global_store_b32 v1, v0, s[2:3]
; CHECK-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-MESA3D-LABEL: test_cluster_id_x:
; CHECK-MESA3D:         .amd_kernel_code_t
; CHECK-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-MESA3D-NEXT:     priority = 0
; CHECK-MESA3D-NEXT:     float_mode = 240
; CHECK-MESA3D-NEXT:     priv = 0
; CHECK-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-MESA3D-NEXT:     debug_mode = 0
; CHECK-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-MESA3D-NEXT:     enable_exception = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-MESA3D-NEXT:     private_element_size = 1
; CHECK-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-MESA3D-NEXT:     wavefront_sgpr_count = 2
; CHECK-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-MESA3D-NEXT:     wavefront_size = 5
; CHECK-MESA3D-NEXT:     call_convention = -1
; CHECK-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-MESA3D-NEXT:  ; %bb.0:
; CHECK-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-MESA3D-NEXT:    v_dual_mov_b32 v0, ttmp9 :: v_dual_mov_b32 v1, 0
; CHECK-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-MESA3D-NEXT:    global_store_b32 v1, v0, s[0:1]
; CHECK-MESA3D-NEXT:    s_endpgm
;
; CHECK-G-UNKNOWN-LABEL: test_cluster_id_x:
; CHECK-G-UNKNOWN:       ; %bb.0:
; CHECK-G-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-G-UNKNOWN-NEXT:    v_dual_mov_b32 v0, ttmp9 :: v_dual_mov_b32 v1, 0
; CHECK-G-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-UNKNOWN-NEXT:    global_store_b32 v1, v0, s[2:3]
; CHECK-G-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-G-MESA3D-LABEL: test_cluster_id_x:
; CHECK-G-MESA3D:         .amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-G-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-G-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-G-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-G-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-G-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-G-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-G-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-G-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     priority = 0
; CHECK-G-MESA3D-NEXT:     float_mode = 240
; CHECK-G-MESA3D-NEXT:     priv = 0
; CHECK-G-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-G-MESA3D-NEXT:     debug_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-G-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-G-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-G-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-G-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-G-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-G-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-G-MESA3D-NEXT:     enable_exception = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-G-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-G-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-G-MESA3D-NEXT:     private_element_size = 1
; CHECK-G-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-G-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-G-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-G-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-G-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-G-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-G-MESA3D-NEXT:     wavefront_sgpr_count = 2
; CHECK-G-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-G-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     wavefront_size = 5
; CHECK-G-MESA3D-NEXT:     call_convention = -1
; CHECK-G-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-G-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:  ; %bb.0:
; CHECK-G-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-G-MESA3D-NEXT:    v_dual_mov_b32 v0, ttmp9 :: v_dual_mov_b32 v1, 0
; CHECK-G-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-MESA3D-NEXT:    global_store_b32 v1, v0, s[0:1]
; CHECK-G-MESA3D-NEXT:    s_endpgm
  %id = call i32 @llvm.amdgcn.cluster.id.x()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_cluster_id_y(ptr addrspace(1) %out) #1 {
; CHECK-UNKNOWN-LABEL: test_cluster_id_y:
; CHECK-UNKNOWN:       ; %bb.0:
; CHECK-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-UNKNOWN-NEXT:    v_dual_mov_b32 v0, ttmp7 :: v_dual_mov_b32 v1, 0
; CHECK-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-UNKNOWN-NEXT:    global_store_b32 v1, v0, s[2:3]
; CHECK-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-MESA3D-LABEL: test_cluster_id_y:
; CHECK-MESA3D:         .amd_kernel_code_t
; CHECK-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-MESA3D-NEXT:     priority = 0
; CHECK-MESA3D-NEXT:     float_mode = 240
; CHECK-MESA3D-NEXT:     priv = 0
; CHECK-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-MESA3D-NEXT:     debug_mode = 0
; CHECK-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-MESA3D-NEXT:     enable_exception = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-MESA3D-NEXT:     private_element_size = 1
; CHECK-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-MESA3D-NEXT:     wavefront_sgpr_count = 2
; CHECK-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-MESA3D-NEXT:     wavefront_size = 5
; CHECK-MESA3D-NEXT:     call_convention = -1
; CHECK-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-MESA3D-NEXT:  ; %bb.0:
; CHECK-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-MESA3D-NEXT:    v_dual_mov_b32 v0, ttmp7 :: v_dual_mov_b32 v1, 0
; CHECK-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-MESA3D-NEXT:    global_store_b32 v1, v0, s[0:1]
; CHECK-MESA3D-NEXT:    s_endpgm
;
; CHECK-G-UNKNOWN-LABEL: test_cluster_id_y:
; CHECK-G-UNKNOWN:       ; %bb.0:
; CHECK-G-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-G-UNKNOWN-NEXT:    v_dual_mov_b32 v0, ttmp7 :: v_dual_mov_b32 v1, 0
; CHECK-G-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-UNKNOWN-NEXT:    global_store_b32 v1, v0, s[2:3]
; CHECK-G-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-G-MESA3D-LABEL: test_cluster_id_y:
; CHECK-G-MESA3D:         .amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-G-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-G-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-G-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-G-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-G-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-G-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-G-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-G-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     priority = 0
; CHECK-G-MESA3D-NEXT:     float_mode = 240
; CHECK-G-MESA3D-NEXT:     priv = 0
; CHECK-G-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-G-MESA3D-NEXT:     debug_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-G-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-G-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-G-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-G-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-G-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-G-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-G-MESA3D-NEXT:     enable_exception = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-G-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-G-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-G-MESA3D-NEXT:     private_element_size = 1
; CHECK-G-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-G-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-G-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-G-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-G-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-G-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-G-MESA3D-NEXT:     wavefront_sgpr_count = 2
; CHECK-G-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-G-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     wavefront_size = 5
; CHECK-G-MESA3D-NEXT:     call_convention = -1
; CHECK-G-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-G-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:  ; %bb.0:
; CHECK-G-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-G-MESA3D-NEXT:    v_dual_mov_b32 v0, ttmp7 :: v_dual_mov_b32 v1, 0
; CHECK-G-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-MESA3D-NEXT:    global_store_b32 v1, v0, s[0:1]
; CHECK-G-MESA3D-NEXT:    s_endpgm
  %id = call i32 @llvm.amdgcn.cluster.id.y()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @test_cluster_id_z(ptr addrspace(1) %out) #1 {
; CHECK-UNKNOWN-LABEL: test_cluster_id_z:
; CHECK-UNKNOWN:       ; %bb.0:
; CHECK-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-UNKNOWN-NEXT:    s_wait_xcnt 0x0
; CHECK-UNKNOWN-NEXT:    s_lshr_b32 s0, ttmp7, 16
; CHECK-UNKNOWN-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; CHECK-UNKNOWN-NEXT:    v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s0
; CHECK-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-UNKNOWN-NEXT:    global_store_b32 v0, v1, s[2:3]
; CHECK-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-MESA3D-LABEL: test_cluster_id_z:
; CHECK-MESA3D:         .amd_kernel_code_t
; CHECK-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-MESA3D-NEXT:     priority = 0
; CHECK-MESA3D-NEXT:     float_mode = 240
; CHECK-MESA3D-NEXT:     priv = 0
; CHECK-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-MESA3D-NEXT:     debug_mode = 0
; CHECK-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-MESA3D-NEXT:     enable_exception = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-MESA3D-NEXT:     private_element_size = 1
; CHECK-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-MESA3D-NEXT:     wavefront_sgpr_count = 3
; CHECK-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-MESA3D-NEXT:     wavefront_size = 5
; CHECK-MESA3D-NEXT:     call_convention = -1
; CHECK-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-MESA3D-NEXT:  ; %bb.0:
; CHECK-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-MESA3D-NEXT:    s_lshr_b32 s2, ttmp7, 16
; CHECK-MESA3D-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; CHECK-MESA3D-NEXT:    v_dual_mov_b32 v0, 0 :: v_dual_mov_b32 v1, s2
; CHECK-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-MESA3D-NEXT:    global_store_b32 v0, v1, s[0:1]
; CHECK-MESA3D-NEXT:    s_endpgm
;
; CHECK-G-UNKNOWN-LABEL: test_cluster_id_z:
; CHECK-G-UNKNOWN:       ; %bb.0:
; CHECK-G-UNKNOWN-NEXT:    s_load_b64 s[2:3], s[0:1], 0x24
; CHECK-G-UNKNOWN-NEXT:    s_wait_xcnt 0x0
; CHECK-G-UNKNOWN-NEXT:    s_lshr_b32 s0, ttmp7, 16
; CHECK-G-UNKNOWN-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; CHECK-G-UNKNOWN-NEXT:    v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, s0
; CHECK-G-UNKNOWN-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-UNKNOWN-NEXT:    global_store_b32 v1, v0, s[2:3]
; CHECK-G-UNKNOWN-NEXT:    s_endpgm
;
; CHECK-G-MESA3D-LABEL: test_cluster_id_z:
; CHECK-G-MESA3D:         .amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:     amd_code_version_major = 1
; CHECK-G-MESA3D-NEXT:     amd_code_version_minor = 2
; CHECK-G-MESA3D-NEXT:     amd_machine_kind = 1
; CHECK-G-MESA3D-NEXT:     amd_machine_version_major = 12
; CHECK-G-MESA3D-NEXT:     amd_machine_version_minor = 5
; CHECK-G-MESA3D-NEXT:     amd_machine_version_stepping = 0
; CHECK-G-MESA3D-NEXT:     kernel_code_entry_byte_offset = 256
; CHECK-G-MESA3D-NEXT:     kernel_code_prefetch_byte_size = 0
; CHECK-G-MESA3D-NEXT:     granulated_workitem_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     granulated_wavefront_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     priority = 0
; CHECK-G-MESA3D-NEXT:     float_mode = 240
; CHECK-G-MESA3D-NEXT:     priv = 0
; CHECK-G-MESA3D-NEXT:     enable_dx10_clamp = 0
; CHECK-G-MESA3D-NEXT:     debug_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_ieee_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_wgp_mode = 0
; CHECK-G-MESA3D-NEXT:     enable_mem_ordered = 1
; CHECK-G-MESA3D-NEXT:     enable_fwd_progress = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_wave_byte_offset = 0
; CHECK-G-MESA3D-NEXT:     user_sgpr_count = 2
; CHECK-G-MESA3D-NEXT:     enable_trap_handler = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_x = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_y = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_id_z = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_workgroup_info = 0
; CHECK-G-MESA3D-NEXT:     enable_vgpr_workitem_id = 0
; CHECK-G-MESA3D-NEXT:     enable_exception_msb = 0
; CHECK-G-MESA3D-NEXT:     granulated_lds_size = 0
; CHECK-G-MESA3D-NEXT:     enable_exception = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_buffer = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_queue_ptr = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_kernarg_segment_ptr = 1
; CHECK-G-MESA3D-NEXT:     enable_sgpr_dispatch_id = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_flat_scratch_init = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_private_segment_size = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_x = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_y = 0
; CHECK-G-MESA3D-NEXT:     enable_sgpr_grid_workgroup_count_z = 0
; CHECK-G-MESA3D-NEXT:     enable_wavefront_size32 = 1
; CHECK-G-MESA3D-NEXT:     enable_ordered_append_gds = 0
; CHECK-G-MESA3D-NEXT:     private_element_size = 1
; CHECK-G-MESA3D-NEXT:     is_ptr64 = 1
; CHECK-G-MESA3D-NEXT:     is_dynamic_callstack = 0
; CHECK-G-MESA3D-NEXT:     is_debug_enabled = 0
; CHECK-G-MESA3D-NEXT:     is_xnack_enabled = 1
; CHECK-G-MESA3D-NEXT:     workitem_private_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     workgroup_group_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     gds_segment_byte_size = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_byte_size = 8
; CHECK-G-MESA3D-NEXT:     workgroup_fbarrier_count = 0
; CHECK-G-MESA3D-NEXT:     wavefront_sgpr_count = 3
; CHECK-G-MESA3D-NEXT:     workitem_vgpr_count = 2
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_vgpr_count = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_first = 0
; CHECK-G-MESA3D-NEXT:     reserved_sgpr_count = 0
; CHECK-G-MESA3D-NEXT:     debug_wavefront_private_segment_offset_sgpr = 0
; CHECK-G-MESA3D-NEXT:     debug_private_segment_buffer_sgpr = 0
; CHECK-G-MESA3D-NEXT:     kernarg_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     group_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     private_segment_alignment = 4
; CHECK-G-MESA3D-NEXT:     wavefront_size = 5
; CHECK-G-MESA3D-NEXT:     call_convention = -1
; CHECK-G-MESA3D-NEXT:     runtime_loader_kernel_symbol = 0
; CHECK-G-MESA3D-NEXT:    .end_amd_kernel_code_t
; CHECK-G-MESA3D-NEXT:  ; %bb.0:
; CHECK-G-MESA3D-NEXT:    s_load_b64 s[0:1], s[0:1], 0x0
; CHECK-G-MESA3D-NEXT:    s_lshr_b32 s2, ttmp7, 16
; CHECK-G-MESA3D-NEXT:    s_delay_alu instid0(SALU_CYCLE_1)
; CHECK-G-MESA3D-NEXT:    v_dual_mov_b32 v1, 0 :: v_dual_mov_b32 v0, s2
; CHECK-G-MESA3D-NEXT:    s_wait_kmcnt 0x0
; CHECK-G-MESA3D-NEXT:    global_store_b32 v1, v0, s[0:1]
; CHECK-G-MESA3D-NEXT:    s_endpgm
  %id = call i32 @llvm.amdgcn.cluster.id.z()
  store i32 %id, ptr addrspace(1) %out
  ret void
}
