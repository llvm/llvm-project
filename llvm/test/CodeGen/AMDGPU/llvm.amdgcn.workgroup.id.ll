; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=HSA -check-prefix=CI-HSA  %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=HSA -check-prefix=VI-HSA  %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=MESA -check-prefix=SI-MESA %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=MESA -check-prefix=VI-MESA %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

; ALL-LABEL {{^}}test_workgroup_id_x:

; HSA: .amd_kernel_code_t
; HSA: compute_pgm_rsrc2_user_sgpr = 6
; HSA: compute_pgm_rsrc2_tgid_x_en = 1
; HSA: compute_pgm_rsrc2_tgid_y_en = 0
; HSA: compute_pgm_rsrc2_tgid_z_en = 0
; HSA: compute_pgm_rsrc2_tg_size_en = 0
; HSA: compute_pgm_rsrc2_tidig_comp_cnt = 0
; HSA: enable_sgpr_grid_workgroup_count_x = 0
; HSA: enable_sgpr_grid_workgroup_count_y = 0
; HSA: enable_sgpr_grid_workgroup_count_z = 0
; HSA: .end_amd_kernel_code_t

; MESA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s2{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s6{{$}}

; ALL-NOT: [[VCOPY]]
; ALL: {{buffer|flat}}_store_dword [[VCOPY]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @test_workgroup_id_x(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL {{^}}test_workgroup_id_y:
; HSA: compute_pgm_rsrc2_user_sgpr = 6
; HSA: compute_pgm_rsrc2_tgid_x_en = 1
; HSA: compute_pgm_rsrc2_tgid_y_en = 1
; HSA: compute_pgm_rsrc2_tgid_z_en = 0
; HSA: compute_pgm_rsrc2_tg_size_en = 0
; HSA: enable_sgpr_grid_workgroup_count_x = 0
; HSA: enable_sgpr_grid_workgroup_count_y = 0
; HSA: enable_sgpr_grid_workgroup_count_z = 0

; MESA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL-NOT: [[VCOPY]]
; ALL: {{buffer|flat}}_store_dword [[VCOPY]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @test_workgroup_id_y(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.y()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL {{^}}test_workgroup_id_z:
; HSA: compute_pgm_rsrc2_user_sgpr = 6
; HSA: compute_pgm_rsrc2_tgid_x_en = 1
; HSA: compute_pgm_rsrc2_tgid_y_en = 0
; HSA: compute_pgm_rsrc2_tgid_z_en = 1
; HSA: compute_pgm_rsrc2_tg_size_en = 0
; HSA: compute_pgm_rsrc2_tidig_comp_cnt = 0
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 0
; HSA: enable_sgpr_queue_ptr = 0
; HSA: enable_sgpr_kernarg_segment_ptr = 1
; HSA: enable_sgpr_dispatch_id = 0
; HSA: enable_sgpr_flat_scratch_init = 0
; HSA: enable_sgpr_private_segment_size = 0
; HSA: enable_sgpr_grid_workgroup_count_x = 0
; HSA: enable_sgpr_grid_workgroup_count_y = 0
; HSA: enable_sgpr_grid_workgroup_count_z = 0

; MESA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL-NOT: [[VCOPY]]
; ALL: {{buffer|flat}}_store_dword [[VCOPY]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @test_workgroup_id_z(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
