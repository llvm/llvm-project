; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-NOHSA,GCN-NOHSA,FUNC %s
; RUN: llc -global-isel -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=SI-NOHSA,GCN-NOHSA,FUNC %s

; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck  --check-prefixes=VI-NOHSA,GCN-NOHSA,FUNC %s
; RUN: llc -global-isel -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck  --check-prefixes=VI-NOHSA,GCN-NOHSA,FUNC %s

; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=EG,FUNC %s

; Legacy intrinsics that just read implicit parameters

; FUNC-LABEL: {{^}}ngroups_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x0
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x0
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].X
define amdgpu_kernel void @ngroups_x (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.x() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x1
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x4
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].Y
define amdgpu_kernel void @ngroups_y (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.y() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}ngroups_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x2
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x8
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].Z
define amdgpu_kernel void @ngroups_z (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.ngroups.z() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x3
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0xc
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[0].W
define amdgpu_kernel void @global_size_x (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.x() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x4
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x10
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].X
define amdgpu_kernel void @global_size_y (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.y() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}global_size_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x5
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x14
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].Y
define amdgpu_kernel void @global_size_z (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.global.size.z() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_x:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x6
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x18
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].Z
define amdgpu_kernel void @local_size_x (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x7
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x1c
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[1].W
define amdgpu_kernel void @local_size_y (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z:
; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x8
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[4:5], 0x20
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]

; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV {{\*? *}}[[VAL]], KC0[2].X
define amdgpu_kernel void @local_size_z (ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, ptr addrspace(1) %out
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

attributes #0 = { readnone }
