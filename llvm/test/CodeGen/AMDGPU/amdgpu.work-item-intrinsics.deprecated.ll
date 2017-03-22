; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SI-NOHSA -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI  -check-prefix=VI-NOHSA -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; Legacy intrinsics that just read implicit parameters

; FUNC-LABEL: {{^}}ngroups_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x0
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].X
define amdgpu_kernel void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].Y
define amdgpu_kernel void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].Z
define amdgpu_kernel void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x3
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xc
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].W
define amdgpu_kernel void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x10
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].X
define amdgpu_kernel void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x5
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x14
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].Y
define amdgpu_kernel void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x6
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x18
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].Z
define amdgpu_kernel void @local_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x7
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1c
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].W
define amdgpu_kernel void @local_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x20
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[2].X
define amdgpu_kernel void @local_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Legacy use of r600 intrinsics by GCN

; The tgid values are stored in sgprs offset by the number of user
; sgprs.

; FUNC-LABEL: {{^}}tgid_x_legacy:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s2{{$}}
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define amdgpu_kernel void @tgid_x_legacy(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_y_legacy:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
define amdgpu_kernel void @tgid_y_legacy(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_z_legacy:
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], s3{{$}}
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; GCN-NOHSA: COMPUTE_PGM_RSRC2:USER_SGPR: 2
; GCN: COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; GCN: COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; GCN: COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
define amdgpu_kernel void @tgid_z_legacy(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 132{{$}}

; FUNC-LABEL: {{^}}tidig_x_legacy:
; GCN-NOHSA: buffer_store_dword v0
define amdgpu_kernel void @tidig_x_legacy(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 2180{{$}}

; FUNC-LABEL: {{^}}tidig_y_legacy:

; GCN-NOHSA: buffer_store_dword v1
define amdgpu_kernel void @tidig_y_legacy(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; GCN-NOHSA: .section .AMDGPU.config
; GCN-NOHSA: .long 47180
; GCN-NOHSA-NEXT: .long 4228{{$}}

; FUNC-LABEL: {{^}}tidig_z_legacy:
; GCN-NOHSA: buffer_store_dword v2
define amdgpu_kernel void @tidig_z_legacy(i32 addrspace(1)* %out) {
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

declare i32 @llvm.r600.read.local.size.x() #0
declare i32 @llvm.r600.read.local.size.y() #0
declare i32 @llvm.r600.read.local.size.z() #0

declare i32 @llvm.r600.read.tgid.x() #0
declare i32 @llvm.r600.read.tgid.y() #0
declare i32 @llvm.r600.read.tgid.z() #0

declare i32 @llvm.r600.read.tidig.x() #0
declare i32 @llvm.r600.read.tidig.y() #0
declare i32 @llvm.r600.read.tidig.z() #0

attributes #0 = { readnone }
