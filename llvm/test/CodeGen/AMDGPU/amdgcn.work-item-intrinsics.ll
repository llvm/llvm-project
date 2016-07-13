; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SI-NOHSA -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI  -check-prefix=VI-NOHSA -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}workdim:

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xb
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2c
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

define void @workdim (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.read.workdim() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The workgroup.id values are stored in sgprs offset by the number of user
; sgprs.

; FUNC-LABEL: {{^}}workgroup_id_x:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s2{{$}}
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @workgroup_id_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workgroup.id.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}workgroup_id_y:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
define void @workgroup_id_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workgroup.id.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}workgroup_id_z:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3{{$}}
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define void @workgroup_id_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workgroup.id.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 132{{$}}

; FUNC-LABEL: {{^}}workitem_id_x:
; GCN-NOHSA: buffer_store_dword v0
define void @workitem_id_x(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 2180{{$}}

; FUNC-LABEL: {{^}}workitem_id_y:

; GCN-NOHSA: buffer_store_dword v1
define void @workitem_id_y(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 4228{{$}}

; FUNC-LABEL: {{^}}workitem_id_z:
; GCN-NOHSA: buffer_store_dword v2
define void @workitem_id_z(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.amdgcn.workitem.id.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workgroup.id.x() #0
declare i32 @llvm.amdgcn.workgroup.id.y() #0
declare i32 @llvm.amdgcn.workgroup.id.z() #0

declare i32 @llvm.amdgcn.workitem.id.x() #0
declare i32 @llvm.amdgcn.workitem.id.y() #0
declare i32 @llvm.amdgcn.workitem.id.z() #0

declare i32 @llvm.amdgcn.read.workdim() #0
