; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=SI,GCN,SI-NOHSA,FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=VI,VI-NOHSA,GCN,FUNC %s
; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck --check-prefixes=EG,FUNC %s


; FUNC-LABEL: {{^}}local_size_x:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[1].Z

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x6
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x18
; CI-HSA: s_load_dword [[XY:s[0-9]+]], s[4:5], 0x1
; VI-HSA: s_load_dword [[XY:s[0-9]+]], s[4:5], 0x4

; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_x(ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.x() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[1].W

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x7
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1c
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_y(ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.y() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].X

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x20
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_z(ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.r600.read.local.size.z() #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_xy:
; SI-NOHSA-DAG: s_load_dwordx2 s[[[X:[0-9]+]]:[[Y:[0-9+]]]], s[0:1], 0x6
; VI-NOHSA-DAG: s_load_dwordx2 s[[[X:[0-9]+]]:[[Y:[0-9+]]]], s[0:1], 0x18
; GCN: s_mul_i32 [[VAL:s[0-9]+]], s[[X]], s[[Y]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_xy(ptr addrspace(1) %out) {
entry:
  %x = call i32 @llvm.r600.read.local.size.x() #0
  %y = call i32 @llvm.r600.read.local.size.y() #0
  %val = mul i32 %x, %y
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_xz:

; SI-NOHSA-DAG: s_load_dword [[X:s[0-9]+]], s[0:1], 0x6
; SI-NOHSA-DAG: s_load_dword [[Z:s[0-9]+]], s[0:1], 0x8
; VI-NOHSA-DAG: s_load_dword [[X:s[0-9]+]], s[0:1], 0x18
; VI-NOHSA-DAG: s_load_dword [[Z:s[0-9]+]], s[0:1], 0x20
; HSA-DAG: s_and_b32 [[X:s[0-9]+]], [[XY]], 0xffff
; GCN: s_mul_i32 [[VAL:s[0-9]+]], [[X]], [[Z]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_xz(ptr addrspace(1) %out) {
entry:
  %x = call i32 @llvm.r600.read.local.size.x() #0
  %z = call i32 @llvm.r600.read.local.size.z() #0
  %val = mul i32 %x, %z
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_yz:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 1

; SI-NOHSA-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s[0:1], 0x7
; VI-NOHSA-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s[0:1], 0x1c
; GCN: s_mul_i32 [[VAL:s[0-9]+]], s[[#LOAD + 0]], s[[#LOAD + 1]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_yz(ptr addrspace(1) %out) {
entry:
  %y = call i32 @llvm.r600.read.local.size.y() #0
  %z = call i32 @llvm.r600.read.local.size.z() #0
  %val = mul i32 %y, %z
  store i32 %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_xyz:
; HSA: enable_sgpr_private_segment_buffer = 1
; HSA: enable_sgpr_dispatch_ptr = 1

; SI-NOHSA-DAG: s_load_dwordx2 s[[[X:[0-9]+]]:[[Y:[0-9]+]]], s[0:1], 0x6
; SI-NOHSA-DAG: s_load_dword s[[Z:[0-9]+]], s[0:1], 0x8
; VI-NOHSA-DAG: s_load_dwordx2 s[[[X:[0-9]+]]:[[Y:[0-9]+]]], s[0:1], 0x18
; VI-NOHSA-DAG: s_load_dword s[[Z:[0-9]+]], s[0:1], 0x20
; GCN: s_mul_i32 [[M:s[0-9]+]], s[[X]], s[[Y]]
; GCN: s_add_i32 [[VAL:s[0-9]+]], [[M]], s[[Z]]
; GCN-DAG: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_xyz(ptr addrspace(1) %out) {
entry:
  %x = call i32 @llvm.r600.read.local.size.x() #0
  %y = call i32 @llvm.r600.read.local.size.y() #0
  %z = call i32 @llvm.r600.read.local.size.z() #0
  %xy = mul i32 %x, %y
  %xyz = add i32 %xy, %z
  store i32 %xyz, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_x_known_bits:
; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x6
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x18
; GCN-NOT: 0xffff
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NEXT: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_x_known_bits(ptr addrspace(1) %out) {
entry:
  %size = call i32 @llvm.r600.read.local.size.x() #0
  %shl = shl i32 %size, 16
  %shr = lshr i32 %shl, 16
  store i32 %shr, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_y_known_bits:
; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x7
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x1c
; GCN-NOT: 0xffff
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NEXT: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_y_known_bits(ptr addrspace(1) %out) {
entry:
  %size = call i32 @llvm.r600.read.local.size.y() #0
  %shl = shl i32 %size, 16
  %shr = lshr i32 %shl, 16
  store i32 %shr, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}local_size_z_known_bits:
; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x8
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x20
; GCN-NOT: 0xffff
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NEXT: buffer_store_dword [[VVAL]]
define amdgpu_kernel void @local_size_z_known_bits(ptr addrspace(1) %out) {
entry:
  %size = call i32 @llvm.r600.read.local.size.z() #0
  %shl = shl i32 %size, 16
  %shr = lshr i32 %shl, 16
  store i32 %shr, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.r600.read.local.size.x() #0
declare i32 @llvm.r600.read.local.size.y() #0
declare i32 @llvm.r600.read.local.size.z() #0

attributes #0 = { nounwind readnone }
