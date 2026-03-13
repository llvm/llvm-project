; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=amdgpu-attributor %s -o %t.bc
; RUN: llc -mtriple=amdgcn -mcpu=tahiti < %t.bc | FileCheck --check-prefixes=ALL,UNKNOWN-OS %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %t.bc | FileCheck --check-prefixes=ALL,UNKNOWN-OS %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tahiti < %t.bc | FileCheck -check-prefixes=ALL,MESA3D %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga < %t.bc | FileCheck -check-prefixes=ALL,MESA3D %s

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

; ALL-LABEL: {{^}}test_workgroup_id_x:

; MESA3D: .amd_kernel_code_t
; MESA3D: user_sgpr_count = 6
; MESA3D: enable_sgpr_workgroup_id_x = 1
; MESA3D: enable_sgpr_workgroup_id_y = 0
; MESA3D: enable_sgpr_workgroup_id_z = 0
; MESA3D: enable_sgpr_workgroup_info = 0
; MESA3D: enable_vgpr_workitem_id = 0
; MESA3D: enable_sgpr_grid_workgroup_count_x = 0
; MESA3D: enable_sgpr_grid_workgroup_count_y = 0
; MESA3D: enable_sgpr_grid_workgroup_count_z = 0
; MESA3D: .end_amd_kernel_code_t

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s2{{$}}
; MESA3D: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s6{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; MESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOMESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 2
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
; MESA3D: user_sgpr_count = 6
; MESA3D: enable_sgpr_workgroup_id_x = 1
; MESA3D: enable_sgpr_workgroup_id_y = 1
; MESA3D: enable_sgpr_workgroup_id_z = 0
; MESA3D: enable_sgpr_workgroup_info = 0
; MESA3D: enable_sgpr_grid_workgroup_count_x = 0
; MESA3D: enable_sgpr_grid_workgroup_count_y = 0
; MESA3D: enable_sgpr_grid_workgroup_count_z = 0

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; MESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOMESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 2
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
; MESA3D: user_sgpr_count = 6
; MESA3D: enable_sgpr_workgroup_id_x = 1
; MESA3D: enable_sgpr_workgroup_id_y = 0
; MESA3D: enable_sgpr_workgroup_id_z = 1
; MESA3D: enable_sgpr_workgroup_info = 0
; MESA3D: enable_vgpr_workitem_id = 0
; MESA3D: enable_sgpr_private_segment_buffer = 1
; MESA3D: enable_sgpr_dispatch_ptr = 0
; MESA3D: enable_sgpr_queue_ptr = 0
; MESA3D: enable_sgpr_kernarg_segment_ptr = 1
; MESA3D: enable_sgpr_dispatch_id = 0
; MESA3D: enable_sgpr_flat_scratch_init = 0
; MESA3D: enable_sgpr_private_segment_size = 0
; MESA3D: enable_sgpr_grid_workgroup_count_x = 0
; MESA3D: enable_sgpr_grid_workgroup_count_y = 0
; MESA3D: enable_sgpr_grid_workgroup_count_z = 0

; UNKNOWN-OS: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s3{{$}}
; HSA: v_mov_b32_e32 [[VCOPY:v[0-9]+]], s7{{$}}

; ALL: {{buffer|flat}}_store_dword {{.*}}[[VCOPY]]

; MESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 6
; ALL-NOMESA3D: COMPUTE_PGM_RSRC2:USER_SGPR: 2
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
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
