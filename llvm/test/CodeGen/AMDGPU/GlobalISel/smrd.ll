; FIXME: Need to add support for mubuf stores to enable this on SI.
; XUN: llc < %s -march=amdgcn -mcpu=tahiti -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefix=SI --check-prefix=GCN --check-prefix=SIVI %s
; RUN: llc < %s -march=amdgcn -mcpu=bonaire -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefix=CI --check-prefix=GCN %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -show-mc-encoding -verify-machineinstrs -global-isel | FileCheck --check-prefix=VI --check-prefix=GCN --check-prefix=SIVI %s

; REQUIRES: global-isel

; SMRD load with an immediate offset.
; GCN-LABEL: {{^}}smrd0:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x1 ; encoding: [0x01
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x4
define amdgpu_kernel void @smrd0(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 1
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

; SMRD load with the largest possible immediate offset.
; GCN-LABEL: {{^}}smrd1:
; SICI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xff ; encoding: [0xff,0x{{[0-9]+[137]}}
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3fc
define amdgpu_kernel void @smrd1(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 255
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

; SMRD load with an offset greater than the largest possible immediate.
; GCN-LABEL: {{^}}smrd2:
; SI: s_movk_i32 s[[OFFSET:[0-9]]], 0x400
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], s[[OFFSET]] ; encoding: [0x0[[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x100
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x400
; GCN: s_endpgm
define amdgpu_kernel void @smrd2(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 256
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

; SMRD load with a 64-bit offset
; GCN-LABEL: {{^}}smrd3:
; FIXME: There are too many copies here because we don't fold immediates
;        through REG_SEQUENCE
; XSI: s_load_dwordx2 s[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0xb ; encoding: [0x0b
; TODO: Add VI checks
; XGCN: s_endpgm
define amdgpu_kernel void @smrd3(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 4294967296 ; 2 ^ 32
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

; SMRD load with the largest possible immediate offset on VI
; GCN-LABEL: {{^}}smrd4:
; SI: s_mov_b32 [[OFFSET:s[0-9]+]], 0xffffc
; SI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x3ffff
; VI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0xffffc
define amdgpu_kernel void @smrd4(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 262143
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

; SMRD load with an offset greater than the largest possible immediate on VI
; GCN-LABEL: {{^}}smrd5:
; SIVI: s_mov_b32 [[OFFSET:s[0-9]+]], 0x100000
; SIVI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], [[OFFSET]]
; CI: s_load_dword s{{[0-9]}}, s[{{[0-9]:[0-9]}}], 0x40000
; GCN: s_endpgm
define amdgpu_kernel void @smrd5(i32 addrspace(4)* %ptr) {
entry:
  %0 = getelementptr i32, i32 addrspace(4)* %ptr, i64 262144
  %1 = load i32, i32 addrspace(4)* %0
  store i32 %1, i32 addrspace(1)* undef
  ret void
}

