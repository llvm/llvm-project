; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}system_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}system_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}system_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out release, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") release, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") release, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_release:
; GFX89-NOT:  s_waitcnt vmcnt(0){{$}}
; GCN:        flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") release, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_seq_cst:
; GFX89-NOT:  s_waitcnt vmcnt(0){{$}}
; GCN:        flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") release, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_0:
; GFX89: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_private_0(
    i32* %in, i32 addrspace(5)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(5)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_1:
; GFX89: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], s{{[0-9]+}} offen glc slc{{$}}
define amdgpu_kernel void @nontemporal_private_1(
    i32* %in, i32 addrspace(5)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(5)* %out, i32 %tid
  store i32 %val, i32 addrspace(5)* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_global_0:
; GFX8:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
; GFX9:  global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc slc{{$}}
define amdgpu_kernel void @nontemporal_global_0(
    i32* %in, i32 addrspace(1)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(1)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_global_1:
; GFX8:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
; GFX9:  global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off glc slc{{$}}
define amdgpu_kernel void @nontemporal_global_1(
    i32* %in, i32 addrspace(1)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(1)* %out, i32 %tid
  store i32 %val, i32 addrspace(1)* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_local_0:
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_local_0(
    i32* %in, i32 addrspace(3)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(3)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_local_1:
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
define amdgpu_kernel void @nontemporal_local_1(
    i32* %in, i32 addrspace(3)* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32* %in, align 4
  %out.gep = getelementptr inbounds i32, i32 addrspace(3)* %out, i32 %tid
  store i32 %val, i32 addrspace(3)* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_flat_0:
; GFX89: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
define amdgpu_kernel void @nontemporal_flat_0(
    i32* %in, i32* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_flat_1:
; GFX89: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
define amdgpu_kernel void @nontemporal_flat_1(
    i32* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32* %in, align 4
  %out.gep = getelementptr inbounds i32, i32* %out, i32 %tid
  store i32 %val, i32* %out.gep, !nontemporal !0
  ret void
}

!0 = !{i32 1}
