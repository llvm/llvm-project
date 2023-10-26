; RUN: llc < %s -march=amdgcn -mcpu=tahiti -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefixes=SI,GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=bonaire -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefixes=CI,GCN,SICIVI %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefixes=VI,GCN,SICIVI %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -show-mc-encoding -verify-machineinstrs -global-isel < %s | FileCheck --check-prefixes=GFX9_10,GCN,VIGFX9_10,SIVIGFX9_10  %s
; RUN: llc -march=amdgcn -mcpu=gfx1010 -show-mc-encoding -verify-machineinstrs -global-isel < %s | FileCheck --check-prefixes=GFX9_10,GCN,VIGFX9_10,SIVIGFX9_10  %s

; SMRD load with an immediate offset.
; GCN-LABEL: {{^}}smrd0:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4
define amdgpu_kernel void @smrd0(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 1
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; SMRD load with the largest possible immediate offset.
; GCN-LABEL: {{^}}smrd1:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff,0x{{[0-9]+[137]}}
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_kernel void @smrd1(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 255
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; SMRD load with an offset greater than the largest possible immediate.
; GCN-LABEL: {{^}}smrd2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
; GCN: s_endpgm
define amdgpu_kernel void @smrd2(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 256
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; SMRD load with a 64-bit offset
; GCN-LABEL: {{^}}smrd3:
; FIXME: There are too many copies here because we don't fold immediates
;        through REG_SEQUENCE
; XSI: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0xb ; encoding: [0x0b
; TODO: Add VI checks
; XGCN: s_endpgm
define amdgpu_kernel void @smrd3(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 4294967296 ; 2 ^ 32
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd4:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
; GFX9_10: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; GFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
define amdgpu_kernel void @smrd4(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 262143
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd5:
; SIVIGFX9_10: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVIGFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_kernel void @smrd5(ptr addrspace(4) %ptr) {
entry:
  %0 = getelementptr i32, ptr addrspace(4) %ptr, i64 262144
  %1 = load i32, ptr addrspace(4) %0
  store i32 %1, ptr addrspace(1) undef
  ret void
}

; GFX9_10 can use a signed immediate byte offset
; GCN-LABEL: {{^}}smrd6:
; SICIVI: s_add_u32 s{{[0-9]}}, s{{[0-9]}}, -4
; SICIVI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x0
; GFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], -0x4
define amdgpu_kernel void @smrd6(ptr addrspace(1) %out, ptr addrspace(4) %ptr) #0 {
entry:
  %tmp = getelementptr i32, ptr addrspace(4) %ptr, i64 -1
  %tmp1 = load i32, ptr addrspace(4) %tmp
  store i32 %tmp1, ptr addrspace(1) %out
  ret void
}

; Don't use a negative SGPR offset
; GCN-LABEL: {{^}}smrd7:
; GCN: s_add_u32 s{{[0-9]}}, s{{[0-9]}}, 0xffe00000
; SICIVI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x0
; GFX9_10: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x0
define amdgpu_kernel void @smrd7(ptr addrspace(1) %out, ptr addrspace(4) %ptr) #0 {
entry:
  %tmp = getelementptr i32, ptr addrspace(4) %ptr, i64 -524288
  %tmp1 = load i32, ptr addrspace(4) %tmp
  store i32 %tmp1, ptr addrspace(1) %out
  ret void
}
