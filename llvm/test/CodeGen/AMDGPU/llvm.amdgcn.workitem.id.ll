; RUN: sed 's/CODE_OBJECT_VERSION/200/g' %s | llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=kaveri -verify-machineinstrs | FileCheck --check-prefixes=ALL,CO-V2,UNPACKED  %s
; RUN: sed 's/CODE_OBJECT_VERSION/200/g' %s | llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=carrizo -mattr=-flat-for-global -verify-machineinstrs | FileCheck --check-prefixes=ALL,CO-V2,UNPACKED  %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn -mcpu=hawaii -verify-machineinstrs | FileCheck --check-prefixes=ALL,MESA,UNPACKED %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck --check-prefixes=ALL,MESA,UNPACKED %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-unknown-mesa3d -mcpu=hawaii -verify-machineinstrs | FileCheck -check-prefixes=ALL,CO-V2,UNPACKED %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs | FileCheck -check-prefixes=ALL,CO-V2,UNPACKED %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a -verify-machineinstrs | FileCheck -check-prefixes=ALL,PACKED-TID %s
; RUN: sed 's/CODE_OBJECT_VERSION/400/g' %s | llc -mtriple=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx1100 -verify-machineinstrs -amdgpu-enable-vopd=0 | FileCheck -check-prefixes=ALL,PACKED-TID %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 132{{$}}

; ALL-LABEL: {{^}}test_workitem_id_x:
; CO-V2: enable_vgpr_workitem_id = 0

; ALL-NOT: v0
; ALL: {{buffer|flat|global}}_store_{{dword|b32}} {{.*}}v0

; PACKED-TID: .amdhsa_system_vgpr_workitem_id 0
define amdgpu_kernel void @test_workitem_id_x(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 2180{{$}}

; ALL-LABEL: {{^}}test_workitem_id_y:
; CO-V2: enable_vgpr_workitem_id = 1
; CO-V2-NOT: v1
; CO-V2: {{buffer|flat}}_store_dword {{.*}}v1

; PACKED-TID: v_bfe_u32 [[ID:v[0-9]+]], v0, 10, 10
; PACKED-TID: {{buffer|flat|global}}_store_{{dword|b32}} {{.*}}[[ID]]
; PACKED-TID: .amdhsa_system_vgpr_workitem_id 1
define amdgpu_kernel void @test_workitem_id_y(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 4228{{$}}

; ALL-LABEL: {{^}}test_workitem_id_z:
; CO-V2: enable_vgpr_workitem_id = 2
; CO-V2-NOT: v2
; CO-V2: {{buffer|flat}}_store_dword {{.*}}v2

; PACKED-TID: v_bfe_u32 [[ID:v[0-9]+]], v0, 20, 10
; PACKED-TID: {{buffer|flat|global}}_store_{{dword|b32}} {{.*}}[[ID]]
; PACKED-TID: .amdhsa_system_vgpr_workitem_id 2
define amdgpu_kernel void @test_workitem_id_z(ptr addrspace(1) %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id, ptr addrspace(1) %out
  ret void
}

; FIXME: Packed tid should avoid the and
; ALL-LABEL: {{^}}test_reqd_workgroup_size_x_only:
; CO-V2: enable_vgpr_workitem_id = 0

; ALL-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; UNPACKED-DAG: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v0

; PACKED: v_and_b32_e32 [[MASKED:v[0-9]+]], 0x3ff, v0
; PACKED: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]

; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]
; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]
define amdgpu_kernel void @test_reqd_workgroup_size_x_only(ptr %out) !reqd_work_group_size !0 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %id.x, ptr %out
  store volatile i32 %id.y, ptr %out
  store volatile i32 %id.z, ptr %out
  ret void
}

; ALL-LABEL: {{^}}test_reqd_workgroup_size_y_only:
; CO-V2: enable_vgpr_workitem_id = 1

; ALL: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]

; UNPACKED: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v1

; PACKED: v_bfe_u32 [[MASKED:v[0-9]+]], v0, 10, 10
; PACKED: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]

; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]
define amdgpu_kernel void @test_reqd_workgroup_size_y_only(ptr %out) !reqd_work_group_size !1 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %id.x, ptr %out
  store volatile i32 %id.y, ptr %out
  store volatile i32 %id.z, ptr %out
  ret void
}

; ALL-LABEL: {{^}}test_reqd_workgroup_size_z_only:
; CO-V2: enable_vgpr_workitem_id = 2

; ALL: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]
; ALL: flat_store_{{dword|b32}} v{{\[[0-9]+:[0-9]+\]}}, [[ZERO]]

; UNPACKED: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, v2

; PACKED: v_bfe_u32 [[MASKED:v[0-9]+]], v0, 10, 20
; PACKED: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[MASKED]]
define amdgpu_kernel void @test_reqd_workgroup_size_z_only(ptr %out) !reqd_work_group_size !2 {
  %id.x = call i32 @llvm.amdgcn.workitem.id.x()
  %id.y = call i32 @llvm.amdgcn.workitem.id.y()
  %id.z = call i32 @llvm.amdgcn.workitem.id.z()
  store volatile i32 %id.x, ptr %out
  store volatile i32 %id.y, ptr %out
  store volatile i32 %id.z, ptr %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }

!llvm.module.flags = !{!3}

!0 = !{i32 64, i32 1, i32 1}
!1 = !{i32 1, i32 64, i32 1}
!2 = !{i32 1, i32 1, i32 64}
!3 = !{i32 1, !"amdgpu_code_object_version", i32 CODE_OBJECT_VERSION}
