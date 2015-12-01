; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SI-NOHSA -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI  -check-prefix=VI-NOHSA -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=HSA -check-prefix=CI-HSA -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=HSA -check-prefix=VI-HSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}ngroups_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].X

; HSA: .amd_kernel_code_t

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

; HSA: .end_amd_kernel_code_t


; GCN-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

define void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Y

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Z

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].W

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x3
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xc
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].X

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x10
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].Y

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x5
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x14
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The tgid values are stored in sgprs offset by the number of user
; sgprs.

; FUNC-LABEL: {{^}}tgid_x:
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

; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s2{{$}}
; HSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s6{{$}}
; GCN: buffer_store_dword [[VVAL]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @tgid_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_y:
; HSA: compute_pgm_rsrc2_user_sgpr = 6
; HSA: compute_pgm_rsrc2_tgid_x_en = 1
; HSA: compute_pgm_rsrc2_tgid_y_en = 1
; HSA: compute_pgm_rsrc2_tgid_z_en = 0
; HSA: compute_pgm_rsrc2_tg_size_en = 0
; HSA: enable_sgpr_grid_workgroup_count_x = 0
; HSA: enable_sgpr_grid_workgroup_count_y = 0
; HSA: enable_sgpr_grid_workgroup_count_z = 0
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3
; GCN-HSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s7
; GCN: buffer_store_dword [[VVAL]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @tgid_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_z:
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

; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s7{{$}}
; GCN: buffer_store_dword [[VVAL]]

; HSA: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @tgid_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 132{{$}}

; FUNC-LABEL: {{^}}tidig_x:
; HSA: compute_pgm_rsrc2_tidig_comp_cnt = 0
; GCN: buffer_store_dword v0
define void @tidig_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 2180{{$}}

; FUNC-LABEL: {{^}}tidig_y:

; HSA: compute_pgm_rsrc2_tidig_comp_cnt = 1
; GCN: buffer_store_dword v1
define void @tidig_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 4228{{$}}

; FUNC-LABEL: {{^}}tidig_z:
; HSA: compute_pgm_rsrc2_tidig_comp_cnt = 2
; GCN: buffer_store_dword v2
define void @tidig_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.r600.read.ngroups.x() #0
declare i32 @llvm.r600.read.ngroups.y() #0
declare i32 @llvm.r600.read.ngroups.z() #0

declare i32 @llvm.r600.read.global.size.x() #0
declare i32 @llvm.r600.read.global.size.y() #0
declare i32 @llvm.r600.read.global.size.z() #0

declare i32 @llvm.r600.read.tgid.x() #0
declare i32 @llvm.r600.read.tgid.y() #0
declare i32 @llvm.r600.read.tgid.z() #0

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0
declare i32 @llvm.r600.read.tidig.z() #0

declare i32 @llvm.AMDGPU.read.workdim() #0

attributes #0 = { readnone }
