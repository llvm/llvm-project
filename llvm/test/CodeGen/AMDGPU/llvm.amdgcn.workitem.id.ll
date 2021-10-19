; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,CO-V2  %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 -mcpu=carrizo -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,CO-V2  %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,MESA %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,MESA %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2 %s
; RUN: llc -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2 %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,PACKED-TID %s

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

; MESA: .section .AMDGPU.config
; MESA: .long 47180
; MESA-NEXT: .long 132{{$}}

; ALL-LABEL: {{^}}test_workitem_id_x:
; CO-V2: enable_vgpr_workitem_id = 0

; ALL-NOT: v0
; ALL: {{buffer|flat|global}}_store_dword {{.*}}v0

; PACKED-TID: .amdhsa_system_vgpr_workitem_id 0
define amdgpu_kernel void @test_workitem_id_x(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %id, i32 addrspace(1)* %out
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
; PACKED-TID: {{buffer|flat|global}}_store_dword {{.*}}[[ID]]
; PACKED-TID: .amdhsa_system_vgpr_workitem_id 1
define amdgpu_kernel void @test_workitem_id_y(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %id, i32 addrspace(1)* %out
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
; PACKED-TID: {{buffer|flat|global}}_store_dword {{.*}}[[ID]]
; PACKED-TID: .amdhsa_system_vgpr_workitem_id 2
define amdgpu_kernel void @test_workitem_id_z(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
