; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mattr=-code-object-v3 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=CO-V2 -check-prefix=CI-HSA  %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mattr=-code-object-v3 -mcpu=carrizo -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=CO-V2 -check-prefix=VI-HSA  %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=UNKNOWN-OS -check-prefix=SI-MESA %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=ALL -check-prefix=UNKNOWN-OS -check-prefix=VI-MESA %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mattr=-code-object-v3 -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2,SI-MESA %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mattr=-code-object-v3 -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2,VI-MESA %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

; ALL-LABEL: {{^}}test_workgroup_id_x:

; CO-V2: .amd_kernel_code_t
; CO-V2: user_sgpr_count = 6
; CO-V2: enable_sgpr_workgroup_id_x = 1
; CO-V2: enable_sgpr_workgroup_id_y = 0
; CO-V2: enable_sgpr_workgroup_id_z = 0
; CO-V2: enable_sgpr_workgroup_info = 0
; CO-V2: enable_vgpr_workitem_id = 0
; CO-V2: enable_sgpr_grid_workgroup_count_x = 0
; CO-V2: enable_sgpr_grid_workgroup_count_y = 0
; CO-V2: enable_sgpr_grid_workgroup_count_z = 0
; CO-V2: .end_amd_kernel_code_t

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s2{{$}}
; CO-V2: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s6{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; CO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOCO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define amdgpu_kernel void @test_workgroup_id_x(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workgroup_id_y:
; CO-V2: user_sgpr_count = 6
; CO-V2: enable_sgpr_workgroup_id_x = 1
; CO-V2: enable_sgpr_workgroup_id_y = 1
; CO-V2: enable_sgpr_workgroup_id_z = 0
; CO-V2: enable_sgpr_workgroup_info = 0
; CO-V2: enable_sgpr_grid_workgroup_count_x = 0
; CO-V2: enable_sgpr_grid_workgroup_count_y = 0
; CO-V2: enable_sgpr_grid_workgroup_count_z = 0

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; CO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOCO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define amdgpu_kernel void @test_workgroup_id_y(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.y()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workgroup_id_z:
; CO-V2: user_sgpr_count = 6
; CO-V2: enable_sgpr_workgroup_id_x = 1
; CO-V2: enable_sgpr_workgroup_id_y = 0
; CO-V2: enable_sgpr_workgroup_id_z = 1
; CO-V2: enable_sgpr_workgroup_info = 0
; CO-V2: enable_vgpr_workitem_id = 0
; CO-V2: enable_sgpr_private_segment_buffer = 1
; CO-V2: enable_sgpr_dispatch_ptr = 0
; CO-V2: enable_sgpr_queue_ptr = 0
; CO-V2: enable_sgpr_kernarg_segment_ptr = 1
; CO-V2: enable_sgpr_dispatch_id = 0
; CO-V2: enable_sgpr_flat_scratch_init = 0
; CO-V2: enable_sgpr_private_segment_size = 0
; CO-V2: enable_sgpr_grid_workgroup_count_x = 0
; CO-V2: enable_sgpr_grid_workgroup_count_y = 0
; CO-V2: enable_sgpr_grid_workgroup_count_z = 0

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; CO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOCO-V2: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; ALL: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; ALL: COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; ALL: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define amdgpu_kernel void @test_workgroup_id_z(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
