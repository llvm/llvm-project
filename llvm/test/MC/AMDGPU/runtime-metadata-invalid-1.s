; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 %s 2>&1 | FileCheck %s
; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj %s 2>&1 | FileCheck %s
; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 %s 2>&1 | FileCheck %s
; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj %s 2>&1 | FileCheck %s
; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %s 2>&1 | FileCheck %s
; RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s 2>&1 | FileCheck %s

; CHECK: error: unknown key 'amd.RandomUnknownKey'

	.text
	.hsa_code_object_version 2,1
	.hsa_code_object_isa 8,0,3,"AMD","AMDGPU"
	.amdgpu_runtime_metadata
---
{ amd.MDVersion: [ 2, 1 ], amd.RandomUnknownKey, amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 16, amd.IsaInfoTotalNumSGPRs: 800, amd.IsaInfoAddressableNumSGPRs: 102, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.Kernels:
  - { amd.KernelName: test, amd.Language: OpenCL C, amd.LanguageVersion: [ 1, 0 ], amd.Args:
      - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 1, amd.ArgValueType: 6, amd.ArgTypeName: 'int*', amd.ArgAddrQual: 1, amd.ArgAccQual: 0 }
      - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
      - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
      - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 } } }
...

	.end_amdgpu_runtime_metadata
	.globl	test
	.p2align	8
	.type	test,@function
	.amdgpu_hsa_kernel test
test:                                   ; @test
	.amd_kernel_code_t
		amd_code_version_major = 1
		amd_code_version_minor = 0
		amd_machine_kind = 1
		amd_machine_version_major = 8
		amd_machine_version_minor = 0
		amd_machine_version_stepping = 3
		kernel_code_entry_byte_offset = 256
		kernel_code_prefetch_byte_size = 0
		max_scratch_backing_memory_byte_size = 0
		granulated_workitem_vgpr_count = 0
		granulated_wavefront_sgpr_count = 0
		priority = 0
		float_mode = 192
		priv = 0
		enable_dx10_clamp = 1
		debug_mode = 0
		enable_ieee_mode = 1
		enable_sgpr_private_segment_wave_byte_offset = 0
		user_sgpr_count = 6
		enable_trap_handler = 1
		enable_sgpr_workgroup_id_x = 1
		enable_sgpr_workgroup_id_y = 0
		enable_sgpr_workgroup_id_z = 0
		enable_sgpr_workgroup_info = 0
		enable_vgpr_workitem_id = 0
		enable_exception_msb = 0
		granulated_lds_size = 0
		enable_exception = 0
		enable_sgpr_private_segment_buffer = 1
		enable_sgpr_dispatch_ptr = 0
		enable_sgpr_queue_ptr = 0
		enable_sgpr_kernarg_segment_ptr = 1
		enable_sgpr_dispatch_id = 0
		enable_sgpr_flat_scratch_init = 0
		enable_sgpr_private_segment_size = 0
		enable_sgpr_grid_workgroup_count_x = 0
		enable_sgpr_grid_workgroup_count_y = 0
		enable_sgpr_grid_workgroup_count_z = 0
		enable_ordered_append_gds = 0
		private_element_size = 1
		is_ptr64 = 1
		is_dynamic_callstack = 0
		is_debug_enabled = 0
		is_xnack_enabled = 0
		workitem_private_segment_byte_size = 0
		workgroup_group_segment_byte_size = 0
		gds_segment_byte_size = 0
		kernarg_segment_byte_size = 8
		workgroup_fbarrier_count = 0
		wavefront_sgpr_count = 6
		workitem_vgpr_count = 3
		reserved_vgpr_first = 0
		reserved_vgpr_count = 0
		reserved_sgpr_first = 0
		reserved_sgpr_count = 0
		debug_wavefront_private_segment_offset_sgpr = 0
		debug_private_segment_buffer_sgpr = 0
		kernarg_segment_alignment = 4
		group_segment_alignment = 4
		private_segment_alignment = 4
		wavefront_size = 6
		call_convention = -1
		runtime_loader_kernel_symbol = 0
	.end_amd_kernel_code_t
; BB#0:                                 ; %entry
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	v_mov_b32_e32 v2, 0x309
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, s0
	v_mov_b32_e32 v1, s1
	flat_store_dword v[0:1], v2
	s_endpgm
.Lfunc_end0:
	.size	test, .Lfunc_end0-test

	.ident	""
	.section	".note.GNU-stack"
