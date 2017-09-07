; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx800 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}nontemporal_load_private_0
; GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_load_private_0(
    i32* %in, i32 addrspace(4)* %out) {
entry:
  %val = load i32, i32* %in, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_private_1
; GCN: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_load_private_1(
    i32* %in, i32 addrspace(4)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32* %in, i32 %tid
  %val = load i32, i32* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_global_0
; GCN: s_load_dword s{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0x0{{$}}
define amdgpu_kernel void @nontemporal_load_global_0(
    i32 addrspace(1)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load i32, i32 addrspace(1)* %in, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_global_1
; GFX8: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX9: global_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], off glc slc{{$}}
define amdgpu_kernel void @nontemporal_load_global_1(
    i32 addrspace(1)* %in, i32 addrspace(4)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_local_0
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_load_local_0(
    i32 addrspace(3)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load i32, i32 addrspace(3)* %in, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_local_1
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_load_local_1(
    i32 addrspace(3)* %in, i32 addrspace(4)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(3)* %in, i32 %tid
  %val = load i32, i32 addrspace(3)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_flat_0
; GCN: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
define amdgpu_kernel void @nontemporal_load_flat_0(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load i32, i32 addrspace(4)* %in, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_load_flat_1
; GCN: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
define amdgpu_kernel void @nontemporal_load_flat_1(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(4)* %in, i32 %tid
  %val = load i32, i32 addrspace(4)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

!0 = !{i32 1}
