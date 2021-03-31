; RUN: llc -global-isel -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,HSA,CO-V2  %s
; RUN: llc -global-isel -mtriple=amdgcn-unknown-amdhsa --amdhsa-code-object-version=2 -mcpu=carrizo -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,HSA,CO-V2  %s
; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=hawaii -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,MESA %s
; RUN: llc -global-isel -mtriple=amdgcn-- -mcpu=tonga -mattr=+flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=ALL,MESA %s
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mattr=+flat-for-global -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2 %s
; RUN: llc -global-isel -mtriple=amdgcn-unknown-mesa3d -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,CO-V2 %s
; RUN: llc -global-isel -march=amdgcn -mtriple=amdgcn-unknown-amdhsa -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=ALL,PACKED-TID %s

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

; PACKED-TID: v_lshrrev_b32_e32 [[ID:v[0-9]+]], 10, v0
; PACKED-TID: v_and_b32_e32 [[ID]], 0x3ff, [[ID]]
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

; PACKED-TID: v_lshrrev_b32_e32 [[ID:v[0-9]+]], 20, v0
; PACKED-TID: v_and_b32_e32 [[ID]], 0x3ff, [[ID]]
; PACKED-TID: {{buffer|flat|global}}_store_dword {{.*}}[[ID]]
; PACKED-TID: .amdhsa_system_vgpr_workitem_id 2
define amdgpu_kernel void @test_workitem_id_z(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workitem_id_x_usex2:
; ALL-NOT: v0
; ALL: {{flat|global}}_store_dword v{{.*}}, v0
; ALL-NOT: v0
; ALL: {{flat|global}}_store_dword v{{.*}}, v0
define amdgpu_kernel void @test_workitem_id_x_usex2(i32 addrspace(1)* %out) #1 {
  %id0 = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id0, i32 addrspace(1)* %out

  %id1 = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id1, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workitem_id_x_use_outside_entry:
; ALL-NOT: v0
; ALL: {{flat|global}}_store_dword
; ALL-NOT: v0
; ALL: {{flat|global}}_store_dword v{{.*}}, v0
define amdgpu_kernel void @test_workitem_id_x_use_outside_entry(i32 addrspace(1)* %out, i32 %arg) #1 {
bb0:
  store volatile i32 0, i32 addrspace(1)* %out
  %cond = icmp eq i32 %arg, 0
  br i1 %cond, label %bb1, label %bb2

bb1:
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  store volatile i32 %id, i32 addrspace(1)* %out
  br label %bb2

bb2:
  ret void
}

; ALL-LABEL: {{^}}test_workitem_id_x_func:
; ALL: s_waitcnt
; HSA-NEXT: v_and_b32_e32 v2, 0x3ff, v31
; MESA-NEXT: v_and_b32_e32 v2, 0x3ff, v2
define void @test_workitem_id_x_func(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workitem_id_y_func:
; HSA: v_lshrrev_b32_e32 v2, 10, v31
; MESA: v_lshrrev_b32_e32 v2, 10, v2
define void @test_workitem_id_y_func(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.y()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

; ALL-LABEL: {{^}}test_workitem_id_z_func:
; HSA: v_lshrrev_b32_e32 v2, 20, v31
; MESA: v_lshrrev_b32_e32 v2, 20, v2
define void @test_workitem_id_z_func(i32 addrspace(1)* %out) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.z()
  store i32 %id, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
