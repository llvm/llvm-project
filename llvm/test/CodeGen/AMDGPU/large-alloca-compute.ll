; RUN: llc -march=amdgcn -mcpu=bonaire < %s | FileCheck -check-prefix=GCN -check-prefix=CI -check-prefix=ALL %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=ALL %s
; RUN: llc -march=amdgcn -mcpu=bonaire -mtriple=amdgcn-unknown-amdhsa < %s -mattr=-flat-for-global | FileCheck -check-prefix=GCNHSA -check-prefix=CIHSA -check-prefix=ALL %s
; RUN: llc -march=amdgcn -mcpu=tonga -mtriple=amdgcn-unknown-amdhsa -mattr=-flat-for-global < %s | FileCheck -check-prefix=GCNHSA -check-prefix=VIHSA -check-prefix=ALL %s

; FIXME: align on alloca seems to be ignored for private_segment_alignment

; ALL-LABEL: {{^}}large_alloca_compute_shader:

; GCN: s_mov_b32 s8, SCRATCH_RSRC_DWORD0
; GCN: s_mov_b32 s9, SCRATCH_RSRC_DWORD1
; GCN: s_mov_b32 s10, -1
; CI: s_mov_b32 s11, 0x80f000
; VI: s_mov_b32 s11, 0x800000


; GCNHSA: .amd_kernel_code_t

; GCNHSA: compute_pgm_rsrc2_scratch_en = 1
; GCNHSA: compute_pgm_rsrc2_user_sgpr = 6
; GCNHSA: compute_pgm_rsrc2_tgid_x_en = 1
; GCNHSA: compute_pgm_rsrc2_tgid_y_en = 0
; GCNHSA: compute_pgm_rsrc2_tgid_z_en = 0
; GCNHSA: compute_pgm_rsrc2_tg_size_en = 0
; GCNHSA: compute_pgm_rsrc2_tidig_comp_cnt = 0

; GCNHSA: enable_sgpr_private_segment_buffer = 1
; GCNHSA: enable_sgpr_dispatch_ptr = 0
; GCNHSA: enable_sgpr_queue_ptr = 0
; GCNHSA: enable_sgpr_kernarg_segment_ptr = 1
; GCNHSA: enable_sgpr_dispatch_id = 0
; GCNHSA: enable_sgpr_flat_scratch_init = 0
; GCNHSA: enable_sgpr_private_segment_size = 0
; GCNHSA: enable_sgpr_grid_workgroup_count_x = 0
; GCNHSA: enable_sgpr_grid_workgroup_count_y = 0
; GCNHSA: enable_sgpr_grid_workgroup_count_z = 0
; GCNHSA: workitem_private_segment_byte_size = 32772
; GCNHSA: private_segment_alignment = 4
; GCNHSA: .end_amd_kernel_code_t


; GCNHSA: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, s[0:3], s7 offen
; GCNHSA: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, s[0:3], s7 offen

; Scratch size = alloca size + emergency stack slot
; ALL: ; ScratchSize: 32772
define void @large_alloca_compute_shader(i32 %x, i32 %y) #0 {
  %large = alloca [8192 x i32], align 4
  %gep = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 8191
  store volatile i32 %x, i32* %gep
  %gep1 = getelementptr [8192 x i32], [8192 x i32]* %large, i32 0, i32 %y
  %val = load volatile i32, i32* %gep1
  store volatile i32 %val, i32 addrspace(1)* undef
  ret void
}

attributes #0 = { nounwind  }
