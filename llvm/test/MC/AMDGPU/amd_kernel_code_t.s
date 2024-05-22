; RUN: llvm-mc -triple=amdgcn-mesa-mesa3d -mcpu=gfx900 -filetype=asm < %s | FileCheck --check-prefix=ASM %s
; RUN: llvm-mc -triple=amdgcn-mesa-mesa3d -mcpu=gfx900 -filetype=obj < %s > %t
; RUN: llvm-objdump -s %t | FileCheck --check-prefix=OBJDUMP %s

; OBJDUMP: Contents of section .known_is_dynamic_callstack:
; OBJDUMP: 0030 00000000 00000000 00001000 00000000

; OBJDUMP: Contents of section .known_wavefront_sgpr_count:
; OBJDUMP: 0050 00000000 01000000 00000000 00000000

; OBJDUMP: Contents of section .known_workitem_vgpr_count:
; OBJDUMP: 0050 00000000 00000100 00000000 00000000

; OBJDUMP: Contents of section .known_workitem_private_segment_byte_size:
; OBJDUMP: 0030 00000000 00000000 00000000 01000000

; OBJDUMP: Contents of section .known_granulated_workitem_vgpr_count:
; OBJDUMP: 0030 01000000 00000000 00000000 00000000

; OBJDUMP: Contents of section .known_enable_sgpr_workgroup_id_x:
; OBJDUMP: 0030 00000000 80000000 00000000 00000000

; OBJDUMP: Contents of section .unknown_is_dynamic_callstack:
; OBJDUMP: 0030 00000000 00000000 00001000 00000000

; OBJDUMP: Contents of section .unknown_wavefront_sgpr_count:
; OBJDUMP: 0050 00000000 01000000 00000000 00000000

; OBJDUMP: Contents of section .unknown_workitem_vgpr_count:
; OBJDUMP: 0050 00000000 00000100 00000000 00000000

; OBJDUMP: Contents of section .unknown_workitem_private_segment_byte_size:
; OBJDUMP: 0030 00000000 00000000 00000000 01000000

; OBJDUMP: Contents of section .unknown_granulated_workitem_vgpr_count:
; OBJDUMP: 0030 01000000 00000000 00000000 00000000

; OBJDUMP: Contents of section .unknown_enable_sgpr_workgroup_id_x:
; OBJDUMP: 0030 00000000 80000000 00000000 00000000

.set known, 1

; ASM-LABEL: known_is_dynamic_callstack:
; ASM: is_dynamic_callstack = 1
.section .known_is_dynamic_callstack
known_is_dynamic_callstack:
	.amd_kernel_code_t
		is_dynamic_callstack = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: known_wavefront_sgpr_count:
; ASM: wavefront_sgpr_count = 1
.section .known_wavefront_sgpr_count
known_wavefront_sgpr_count:
	.amd_kernel_code_t
		wavefront_sgpr_count = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: known_workitem_vgpr_count:
; ASM: workitem_vgpr_count = 1
.section .known_workitem_vgpr_count
known_workitem_vgpr_count:
	.amd_kernel_code_t
		workitem_vgpr_count = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: known_workitem_private_segment_byte_size:
; ASM: workitem_private_segment_byte_size = 1
.section .known_workitem_private_segment_byte_size
known_workitem_private_segment_byte_size:
	.amd_kernel_code_t
		workitem_private_segment_byte_size = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: known_granulated_workitem_vgpr_count:
; ASM: granulated_workitem_vgpr_count = 1
.section .known_granulated_workitem_vgpr_count
known_granulated_workitem_vgpr_count:
	.amd_kernel_code_t
		granulated_workitem_vgpr_count = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: known_enable_sgpr_workgroup_id_x:
; ASM: enable_sgpr_workgroup_id_x = 1
.section .known_enable_sgpr_workgroup_id_x
known_enable_sgpr_workgroup_id_x:
	.amd_kernel_code_t
		enable_sgpr_workgroup_id_x = known
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_is_dynamic_callstack:
; ASM: is_dynamic_callstack = unknown
.section .unknown_is_dynamic_callstack
unknown_is_dynamic_callstack:
	.amd_kernel_code_t
		is_dynamic_callstack = unknown
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_wavefront_sgpr_count:
; ASM: wavefront_sgpr_count = unknown
.section .unknown_wavefront_sgpr_count
unknown_wavefront_sgpr_count:
	.amd_kernel_code_t
		wavefront_sgpr_count = unknown
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_workitem_vgpr_count:
; ASM: workitem_vgpr_count = unknown
.section .unknown_workitem_vgpr_count
unknown_workitem_vgpr_count:
	.amd_kernel_code_t
		workitem_vgpr_count = unknown
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_workitem_private_segment_byte_size:
; ASM: workitem_private_segment_byte_size = unknown
.section .unknown_workitem_private_segment_byte_size
unknown_workitem_private_segment_byte_size:
	.amd_kernel_code_t
		workitem_private_segment_byte_size = unknown
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_granulated_workitem_vgpr_count:
; ASM: granulated_workitem_vgpr_count = ((0&4294967232)|(unknown&63))&63
; ASM: granulated_wavefront_sgpr_count = (((0&4294967232)|(unknown&63))>>6)&15
; ASM: priority = (((0&4294967232)|(unknown&63))>>10)&3
; ASM: float_mode = (((0&4294967232)|(unknown&63))>>12)&255
; ASM: priv = (((0&4294967232)|(unknown&63))>>20)&1
; ASM: enable_dx10_clamp = (((0&4294967232)|(unknown&63))>>21)&1
; ASM: debug_mode = (((0&4294967232)|(unknown&63))>>22)&1
; ASM: enable_ieee_mode = (((0&4294967232)|(unknown&63))>>23)&1
; ASM: enable_wgp_mode = (((0&4294967232)|(unknown&63))>>29)&1
; ASM: enable_mem_ordered = (((0&4294967232)|(unknown&63))>>30)&1
; ASM: enable_fwd_progress = (((0&4294967232)|(unknown&63))>>31)&1
.section .unknown_granulated_workitem_vgpr_count
unknown_granulated_workitem_vgpr_count:
	.amd_kernel_code_t
		granulated_workitem_vgpr_count = unknown
	.end_amd_kernel_code_t
	s_endpgm

; ASM-LABEL: unknown_enable_sgpr_workgroup_id_x:
; ASM: enable_sgpr_private_segment_wave_byte_offset = ((0&4294967167)|((unknown&1)<<7))&1
; ASM: user_sgpr_count = (((0&4294967167)|((unknown&1)<<7))>>1)&31
; ASM: enable_trap_handler = (((0&4294967167)|((unknown&1)<<7))>>6)&1
; ASM: enable_sgpr_workgroup_id_x = (((0&4294967167)|((unknown&1)<<7))>>7)&1
; ASM: enable_sgpr_workgroup_id_y = (((0&4294967167)|((unknown&1)<<7))>>8)&1
; ASM: enable_sgpr_workgroup_id_z = (((0&4294967167)|((unknown&1)<<7))>>9)&1
; ASM: enable_sgpr_workgroup_info = (((0&4294967167)|((unknown&1)<<7))>>10)&1
; ASM: enable_vgpr_workitem_id = (((0&4294967167)|((unknown&1)<<7))>>11)&3
; ASM: enable_exception_msb = (((0&4294967167)|((unknown&1)<<7))>>13)&3
; ASM: granulated_lds_size = (((0&4294967167)|((unknown&1)<<7))>>15)&511
; ASM: enable_exception = (((0&4294967167)|((unknown&1)<<7))>>24)&127
.section .unknown_enable_sgpr_workgroup_id_x
unknown_enable_sgpr_workgroup_id_x:
	.amd_kernel_code_t
		enable_sgpr_workgroup_id_x = unknown
	.end_amd_kernel_code_t
	s_endpgm

.set unknown, 1
