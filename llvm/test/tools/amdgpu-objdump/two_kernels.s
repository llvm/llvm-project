// RUN: llvm-mc -arch=amdgcn -filetype=obj %s | amdgpu-objdump - | FileCheck %s --check-prefix=DIS

.text

.hsa_code_object_version 2,0

.hsa_code_object_isa 8,0,0,"AMD","AMDGPU"

.amdgpu_hsa_kernel amd_kernel_code_t_test_all
.amdgpu_hsa_kernel amd_kernel_code_t_minimal


amd_kernel_code_t_test_all:
.amd_kernel_code_t
    kernel_code_version_major = 100
    kernel_code_version_minor = 100
    machine_kind = 0
    machine_version_major = 5
    machine_version_minor = 5
    machine_version_stepping = 5
    kernel_code_entry_byte_offset = 512
    kernel_code_prefetch_byte_size = 1
    max_scratch_backing_memory_byte_size = 1
    compute_pgm_rsrc1_vgprs = 1
    compute_pgm_rsrc1_sgprs = 1
    compute_pgm_rsrc1_priority = 1
    compute_pgm_rsrc1_float_mode = 1
    compute_pgm_rsrc1_priv = 1
    compute_pgm_rsrc1_dx10_clamp = 1
    compute_pgm_rsrc1_debug_mode = 1
    compute_pgm_rsrc1_ieee_mode = 1
    compute_pgm_rsrc2_scratch_en = 1
    compute_pgm_rsrc2_user_sgpr = 1
    compute_pgm_rsrc2_tgid_x_en = 1
    compute_pgm_rsrc2_tgid_y_en = 1
    compute_pgm_rsrc2_tgid_z_en = 1
    compute_pgm_rsrc2_tg_size_en = 1
    compute_pgm_rsrc2_tidig_comp_cnt = 1
    compute_pgm_rsrc2_excp_en_msb = 1
    compute_pgm_rsrc2_lds_size = 1
    compute_pgm_rsrc2_excp_en = 1
    enable_sgpr_private_segment_buffer = 1
    enable_sgpr_dispatch_ptr = 1
    enable_sgpr_queue_ptr = 1
    enable_sgpr_kernarg_segment_ptr = 1
    enable_sgpr_dispatch_id = 1
    enable_sgpr_flat_scratch_init = 1
    enable_sgpr_private_segment_size = 1
    enable_sgpr_grid_workgroup_count_x = 1
    enable_sgpr_grid_workgroup_count_y = 1
    enable_sgpr_grid_workgroup_count_z = 1
    enable_ordered_append_gds = 1
    private_element_size = 1
    is_ptr64 = 1
    is_dynamic_callstack = 1
    is_debug_enabled = 1
    is_xnack_enabled = 1
    workitem_private_segment_byte_size = 1
    workgroup_group_segment_byte_size = 1
    gds_segment_byte_size = 1
    kernarg_segment_byte_size = 1
    workgroup_fbarrier_count = 1
    wavefront_sgpr_count = 1
    workitem_vgpr_count = 1
    reserved_vgpr_first = 1
    reserved_vgpr_count = 1
    reserved_sgpr_first = 1
    reserved_sgpr_count = 1
    debug_wavefront_private_segment_offset_sgpr = 1
    debug_private_segment_buffer_sgpr = 1
    kernarg_segment_alignment = 5
    group_segment_alignment = 5
    private_segment_alignment = 5
    wavefront_size = 5
    call_convention = 1
    runtime_loader_kernel_symbol = 1
.end_amd_kernel_code_t


amd_kernel_code_t_minimal:
.amd_kernel_code_t
    enable_sgpr_kernarg_segment_ptr = 1
    is_ptr64 = 1
    granulated_workitem_vgpr_count = 1
    granulated_wavefront_sgpr_count = 1
    user_sgpr_count = 2
    kernarg_segment_byte_size = 16
    wavefront_sgpr_count = 8
    workitem_vgpr_count = 16
.end_amd_kernel_code_t


// DIS: .hsa_code_object_version 2,0
// DIS: .hsa_code_object_isa 8,0,0,"AMD","AMDGPU"

// DIS: .amdgpu_hsa_kernel amd_kernel_code_t_minimal

// DIS: amd_kernel_code_t_minimal:
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
// DIS:     granulated_workitem_vgpr_count = 1
// DIS:     granulated_wavefront_sgpr_count = 1
// DIS:     priority = 0
// DIS:     float_mode = 0
// DIS:     priv = 0
// DIS:     enable_dx10_clamp = 0
// DIS:     debug_mode = 0
// DIS:     enable_ieee_mode = 0
// DIS:     enable_sgpr_private_segment_wave_byte_offset = 0
// DIS:     user_sgpr_count = 2
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
// DIS:     enable_sgpr_kernarg_segment_ptr = 1
// DIS:     enable_sgpr_dispatch_id = 0
// DIS:     enable_sgpr_flat_scratch_init = 0
// DIS:     enable_sgpr_private_segment_size = 0
// DIS:     enable_sgpr_grid_workgroup_count_x = 0
// DIS:     enable_sgpr_grid_workgroup_count_y = 0
// DIS:     enable_sgpr_grid_workgroup_count_z = 0
// DIS:     enable_ordered_append_gds = 0
// DIS:     private_element_size = 0
// DIS:     is_ptr64 = 1
// DIS:     is_dynamic_callstack = 0
// DIS:     is_debug_enabled = 0
// DIS:     is_xnack_enabled = 0
// DIS:     workitem_private_segment_byte_size = 0
// DIS:     workgroup_group_segment_byte_size = 0
// DIS:     gds_segment_byte_size = 0
// DIS:     kernarg_segment_byte_size = 16
// DIS:     workgroup_fbarrier_count = 0
// DIS:     wavefront_sgpr_count = 8
// DIS:     workitem_vgpr_count = 16
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

// DIS: .amdgpu_hsa_kernel amd_kernel_code_t_test_all

// DIS: amd_kernel_code_t_test_all:
// DIS: .amd_kernel_code_t
// DIS:     amd_code_version_major = 100
// DIS:     amd_code_version_minor = 100
// DIS:     amd_machine_kind = 0
// DIS:     amd_machine_version_major = 5
// DIS:     amd_machine_version_minor = 5
// DIS:     amd_machine_version_stepping = 5
// DIS:     kernel_code_entry_byte_offset = 512
// DIS:     kernel_code_prefetch_byte_size = 1
// DIS:     max_scratch_backing_memory_byte_size = 1
// DIS:     granulated_workitem_vgpr_count = 1
// DIS:     granulated_wavefront_sgpr_count = 1
// DIS:     priority = 1
// DIS:     float_mode = 1
// DIS:     priv = 1
// DIS:     enable_dx10_clamp = 1
// DIS:     debug_mode = 1
// DIS:     enable_ieee_mode = 1
// DIS:     enable_sgpr_private_segment_wave_byte_offset = 1
// DIS:     user_sgpr_count = 1
// DIS:     enable_sgpr_workgroup_id_x = 1
// DIS:     enable_sgpr_workgroup_id_y = 1
// DIS:     enable_sgpr_workgroup_id_z = 1
// DIS:     enable_sgpr_workgroup_info = 1
// DIS:     enable_vgpr_workitem_id = 1
// DIS:     enable_exception_msb = 1
// DIS:     granulated_lds_size = 1
// DIS:     enable_exception = 1
// DIS:     enable_sgpr_private_segment_buffer = 1
// DIS:     enable_sgpr_dispatch_ptr = 1
// DIS:     enable_sgpr_queue_ptr = 1
// DIS:     enable_sgpr_kernarg_segment_ptr = 1
// DIS:     enable_sgpr_dispatch_id = 1
// DIS:     enable_sgpr_flat_scratch_init = 1
// DIS:     enable_sgpr_private_segment_size = 1
// DIS:     enable_sgpr_grid_workgroup_count_x = 1
// DIS:     enable_sgpr_grid_workgroup_count_y = 1
// DIS:     enable_sgpr_grid_workgroup_count_z = 1
// DIS:     enable_ordered_append_gds = 1
// DIS:     private_element_size = 1
// DIS:     is_ptr64 = 1
// DIS:     is_dynamic_callstack = 1
// DIS:     is_debug_enabled = 1
// DIS:     is_xnack_enabled = 1
// DIS:     workitem_private_segment_byte_size = 1
// DIS:     workgroup_group_segment_byte_size = 1
// DIS:     gds_segment_byte_size = 1
// DIS:     kernarg_segment_byte_size = 1
// DIS:     workgroup_fbarrier_count = 1
// DIS:     wavefront_sgpr_count = 1
// DIS:     workitem_vgpr_count = 1
// DIS:     reserved_vgpr_first = 1
// DIS:     reserved_vgpr_count = 1
// DIS:     reserved_sgpr_first = 1
// DIS:     reserved_sgpr_count = 1
// DIS:     debug_wavefront_private_segment_offset_sgpr = 1
// DIS:     debug_private_segment_buffer_sgpr = 1
// DIS:     kernarg_segment_alignment = 5
// DIS:     group_segment_alignment = 5
// DIS:     private_segment_alignment = 5
// DIS:     wavefront_size = 5
// DIS:     call_convention = 1
// DIS:     runtime_loader_kernel_symbol = 1
// DIS: .end_amd_kernel_code_t
