; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx800 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX8 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefix=GCN --check-prefix=GFX9 %s

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}nontemporal_store_private_0
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_private_0(
    i32 addrspace(4)* %in, i32* %out) {
entry:
  %val = load i32, i32 addrspace(4)* %in, align 4
  store i32 %val, i32* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_private_1
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_private_1(
    i32 addrspace(4)* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32 addrspace(4)* %in, align 4
  %out.gep = getelementptr inbounds i32, i32* %out, i32 %tid
  store i32 %val, i32* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_global_0
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[{{[0-9]+}}:{{[0-9]+}}], 0 glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_global_0(
    i32 addrspace(4)* %in, i32 addrspace(1)* %out) {
entry:
  %val = load i32, i32 addrspace(4)* %in, align 4
  store i32 %val, i32 addrspace(1)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_global_1
; GFX8: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
; GFX9: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_global_1(
    i32 addrspace(4)* %in, i32 addrspace(1)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32 addrspace(4)* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  store i32 %val, i32 addrspace(1)* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_local_0
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_store_local_0(
    i32 addrspace(4)* %in, i32 addrspace(3)* %out) {
entry:
  %val = load i32, i32 addrspace(4)* %in, align 4
  store i32 %val, i32 addrspace(3)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_local_1
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_store_local_1(
    i32 addrspace(4)* %in, i32 addrspace(3)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32 addrspace(4)* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(3)* %out, i32 %tid
  store i32 %val, i32 addrspace(3)* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_flat_0
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_flat_0(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load i32, i32 addrspace(4)* %in, align 4
  store i32 %val, i32 addrspace(4)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_store_flat_1
; GCN: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
define amdgpu_kernel void @nontemporal_store_flat_1(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32 addrspace(4)* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(4)* %out, i32 %tid
  store i32 %val, i32 addrspace(4)* %out.gep, !nontemporal !0
  ret void
}

!0 = !{i32 1}
