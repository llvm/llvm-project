; RUN: llc -march=amdgcn -mcpu=bonaire -show-mc-encoding < %s | FileCheck --check-prefixes=GCN,CI,ALL %s
; RUN: llc -march=amdgcn -mcpu=carrizo --show-mc-encoding < %s | FileCheck --check-prefixes=GCN,VI,ALL %s
; RUN: llc -march=amdgcn -mcpu=gfx900 --show-mc-encoding < %s | FileCheck --check-prefixes=GCN,GFX9,ALL %s
; RUN: llc -march=amdgcn -mcpu=bonaire -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=4 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCNHSA,ALL %s
; RUN: llc -march=amdgcn -mcpu=carrizo -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=4 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCNHSA,ALL %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=4 -mattr=-flat-for-global < %s | FileCheck --check-prefixes=GCNHSA,ALL %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=4 -mattr=-flat-for-global,-architected-flat-scratch,-user-sgpr-init16-bug < %s | FileCheck --check-prefixes=GCNHSA,ALL %s

; FIXME: align on alloca seems to be ignored for private_segment_alignment

; ALL-LABEL: {{^}}large_alloca_compute_shader:

; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG: ; fixup A - offset: 4, value: SCRATCH_RSRC_DWORD0
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN-DAG: ; fixup A - offset: 4, value: SCRATCH_RSRC_DWORD1
; GCN-DAG: s_mov_b32 s{{[0-9]+}}, -1
; CI-DAG: s_mov_b32 s{{[0-9]+}}, 0xe8f000
; VI-DAG: s_mov_b32 s{{[0-9]+}}, 0xe80000
; GFX9-DAG: s_mov_b32 s{{[0-9]+}}, 0xe00000

; GCNHSA: buffer_store_{{dword|b32}} {{v[0-9]+}}, {{v[0-9]+}}, s[0:3], 0 offen
; GCNHSA: buffer_load_{{dword|b32}} {{v[0-9]+}}, {{v[0-9]+}}, s[0:3], 0 offen

; GCNHSA: .amdhsa_kernel large_alloca_compute_shader
; GCNHSA:         .amdhsa_group_segment_fixed_size 0
; GCNHSA:         .amdhsa_private_segment_fixed_size 32772
; GCNHSA:         .amdhsa_user_sgpr_private_segment_buffer 1
; GCNHSA:         .amdhsa_user_sgpr_dispatch_ptr 0
; GCNHSA:         .amdhsa_user_sgpr_queue_ptr 0
; GCNHSA:         .amdhsa_user_sgpr_kernarg_segment_ptr 1
; GCNHSA:         .amdhsa_user_sgpr_dispatch_id 0
; GCNHSA:         .amdhsa_user_sgpr_flat_scratch_init 1
; GCNHSA:         .amdhsa_user_sgpr_private_segment_size 0
; GCNHSA:         .amdhsa_system_sgpr_private_segment_wavefront_offset 1
; GCNHSA:         .amdhsa_system_sgpr_workgroup_id_x 1
; GCNHSA:         .amdhsa_system_sgpr_workgroup_id_y 0
; GCNHSA:         .amdhsa_system_sgpr_workgroup_id_z 0
; GCNHSA:         .amdhsa_system_sgpr_workgroup_info 0
; GCNHSA:         .amdhsa_system_vgpr_workitem_id 0
; GCNHSA:         .amdhsa_next_free_vgpr 3
; GCNHSA:         .amdhsa_next_free_sgpr 10
; GCNHSA:         .amdhsa_float_round_mode_32 0
; GCNHSA:         .amdhsa_float_round_mode_16_64 0
; GCNHSA:         .amdhsa_float_denorm_mode_32 3
; GCNHSA:         .amdhsa_float_denorm_mode_16_64 3
; GCNHSA:         .amdhsa_dx10_clamp 1
; GCNHSA:         .amdhsa_ieee_mode 1
; GCNHSA:         .amdhsa_exception_fp_ieee_invalid_op 0
; GCNHSA:         .amdhsa_exception_fp_denorm_src 0
; GCNHSA:         .amdhsa_exception_fp_ieee_div_zero 0
; GCNHSA:         .amdhsa_exception_fp_ieee_overflow 0
; GCNHSA:         .amdhsa_exception_fp_ieee_underflow 0
; GCNHSA:         .amdhsa_exception_fp_ieee_inexact 0
; GCNHSA:         .amdhsa_exception_int_div_zero 0
; GCNHSA: .end_amdhsa_kernel

; Scratch size = alloca size + emergency stack slot, align {{.*}}, addrspace(5)
; ALL: ; ScratchSize: 32772
define amdgpu_kernel void @large_alloca_compute_shader(i32 %x, i32 %y) #0 {
  %large = alloca [8192 x i32], align 4, addrspace(5)
  %gep = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %large, i32 0, i32 8191
  store volatile i32 %x, i32 addrspace(5)* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32] addrspace(5)* %large, i32 0, i32 %y
  %val = load volatile i32, i32 addrspace(5)* %gep1
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind  }
