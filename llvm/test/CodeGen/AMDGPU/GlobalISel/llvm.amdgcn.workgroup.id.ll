; RUN: sed 's/CODE_OBJECT_VERSION/200/g' %s | llc -global-isel -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck --check-prefixes=ALL,CO-V2  %s
; RUN: sed 's/CODE_OBJECT_VERSION/200/g' %s | llc -global-isel -mtriple=amdgcn-unknown-amdhsa -mcpu=carrizo -verify-machineinstrs | FileCheck --check-prefixes=ALL,CO-V2  %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -global-isel -mtriple=amdgcn-- -mcpu=hawaii -verify-machineinstrs | FileCheck --check-prefixes=ALL,UNKNOWN-OS %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -global-isel -mtriple=amdgcn-- -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefixes=ALL,UNKNOWN-OS %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mcpu=hawaii -verify-machineinstrs | FileCheck -check-prefixes=ALL,CO-V2 %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga -verify-machineinstrs | FileCheck -check-prefixes=ALL,CO-V2 %s

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
define amdgpu_kernel void @test_workgroup_id_x(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.x()
  store i32 %id, ptr addrspace(1) %out
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
define amdgpu_kernel void @test_workgroup_id_y(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.y()
  store i32 %id, ptr addrspace(1) %out
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
define amdgpu_kernel void @test_workgroup_id_z(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workgroup.id.z()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 CODE_OBJECT_VERSION}
