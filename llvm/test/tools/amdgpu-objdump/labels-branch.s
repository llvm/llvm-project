// RUN: llvm-mc -arch=amdgcn -filetype=obj %s | amdgpu-objdump - | FileCheck %s --check-prefix=DIS

.hsa_code_object_version 2,0

.hsa_code_object_isa 8,0,1,"AMD","AMDGPU"

.amdgpu_hsa_kernel test

test:
.amd_kernel_code_t
.end_amd_kernel_code_t

loop_start:
    s_branch loop_start
    s_branch loop_end
loop_end:
    s_branch gds
gds:
    s_nop 0

// DIS: .hsa_code_object_version 2,0
// DIS: .hsa_code_object_isa 8,0,1,"AMD","AMDGPU"
// DIS: .amdgpu_hsa_kernel test

// DIS: test:
// DIS: .amd_kernel_code_t
// DIS:     amd_code_version_major = 1
// DIS:     amd_code_version_minor = 1
// DIS:     amd_machine_kind = 1
// DIS:     amd_machine_version_major = 0
// DIS:     amd_machine_version_minor = 0
// DIS:     amd_machine_version_stepping = 0
// DIS:     kernel_code_entry_byte_offset = 256
// DIS:     kernel_code_prefetch_byte_size = 0
// DIS:     max_scratch_backing_memory_byte_size = 0
// DIS:     granulated_workitem_vgpr_count = 0
// DIS:     granulated_wavefront_sgpr_count = 0
// DIS:     priority = 0
// DIS:     float_mode = 0
// DIS:     priv = 0
// DIS:     enable_dx10_clamp = 0
// DIS:     debug_mode = 0
// DIS:     enable_ieee_mode = 0
// DIS:     enable_sgpr_private_segment_wave_byte_offset = 0
// DIS:     user_sgpr_count = 0
// DIS:     enable_sgpr_workgroup_id_x = 0
// DIS:     enable_sgpr_workgroup_id_y = 0
// DIS:     enable_sgpr_workgroup_id_z = 0
// DIS:     enable_sgpr_workgroup_info = 0
// DIS:     enable_vgpr_workitem_id = 0
// DIS:     enable_exception_msb = 0
// DIS:     granulated_lds_size = 0
// DIS:     enable_exception = 0
// DIS:     enable_sgpr_private_segment_buffer = 0
// DIS:     enable_sgpr_dispatch_ptr = 0
// DIS:     enable_sgpr_queue_ptr = 0
// DIS:     enable_sgpr_kernarg_segment_ptr = 0
// DIS:     enable_sgpr_dispatch_id = 0
// DIS:     enable_sgpr_flat_scratch_init = 0
// DIS:     enable_sgpr_private_segment_size = 0
// DIS:     enable_sgpr_grid_workgroup_count_x = 0
// DIS:     enable_sgpr_grid_workgroup_count_y = 0
// DIS:     enable_sgpr_grid_workgroup_count_z = 0
// DIS:     enable_ordered_append_gds = 0
// DIS:     private_element_size = 0
// DIS:     is_ptr64 = 0
// DIS:     is_dynamic_callstack = 0
// DIS:     is_debug_enabled = 0
// DIS:     is_xnack_enabled = 0
// DIS:     workitem_private_segment_byte_size = 0
// DIS:     workgroup_group_segment_byte_size = 0
// DIS:     gds_segment_byte_size = 0
// DIS:     kernarg_segment_byte_size = 0
// DIS:     workgroup_fbarrier_count = 0
// DIS:     wavefront_sgpr_count = 0
// DIS:     workitem_vgpr_count = 0
// DIS:     reserved_vgpr_first = 0
// DIS:     reserved_vgpr_count = 0
// DIS:     reserved_sgpr_first = 0
// DIS:     reserved_sgpr_count = 0
// DIS:     debug_wavefront_private_segment_offset_sgpr = 0
// DIS:     debug_private_segment_buffer_sgpr = 0
// DIS:     kernarg_segment_alignment = 4
// DIS:     group_segment_alignment = 4
// DIS:     private_segment_alignment = 4
// DIS:     wavefront_size = 6
// DIS:     call_convention = -1
// DIS:     runtime_loader_kernel_symbol = 0
// DIS: .end_amd_kernel_code_t

// DIS: // Disassembly:
// DIS: loop_start:
// DIS-NEXT:     s_branch loop_start // 000000000100: BF82FFFF
// DIS:     s_branch loop_end // 000000000104: BF820000
// DIS: loop_end:
// DIS:     s_branch gds // 000000000108: BF820000
// DIS: gds:
// DIS:     s_nop 0 // 00000000010C: BF800000
