; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}ngroups_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].X

; GCN: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @ngroups_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Y

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @ngroups_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].Z

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @ngroups_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[0].W

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x3
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xc
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @global_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].X

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x4
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x10
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @global_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].Y

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x5
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x14
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @global_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].Z

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x6
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x18
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @local_size_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[1].W

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x7
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1c
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @local_size_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[2].X

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x20
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @local_size_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}get_work_dim:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[2].Z

; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xb
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2c
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @get_work_dim (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.AMDGPU.read.workdim() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; The tgid values are stored in sgprs offset by the number of user sgprs.
; Currently we always use exactly 2 user sgprs for the pointer to the
; kernel arguments, but this may change in the future.

; FUNC-LABEL: {{^}}tgid_x:
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], s4
; GCN: buffer_store_dword [[VVAL]]
define void @tgid_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_y:
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], s5
; GCN: buffer_store_dword [[VVAL]]
define void @tgid_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tgid_z:
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], s6
; GCN: buffer_store_dword [[VVAL]]
define void @tgid_z (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tgid.z() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_x:
; GCN: buffer_store_dword v0
define void @tidig_x (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.x() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_y:
; GCN: buffer_store_dword v1
define void @tidig_y (i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.r600.read.tidig.y() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}tidig_z:
; GCN: buffer_store_dword v2
define void @tidig_z (i32 addrspace(1)* %out) {
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

declare i32 @llvm.AMDGPU.read.workdim() #0

attributes #0 = { readnone }
