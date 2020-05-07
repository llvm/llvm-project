; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx900 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3 -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10WGP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3,+cumode -verify-machineinstrs -amdgpu-enable-global-sgpr-addr < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10CU %s

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}system_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("one-as") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("one-as") release, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("one-as") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread-one-as") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread-one-as") release, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread-one-as") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent-one-as") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent-one-as") release, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent-one-as") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup-one-as") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_release:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup-one-as") release, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup-one-as") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront-one-as") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront-one-as") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront-one-as") release, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront-one-as") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_0:
; GFX89: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen glc slc{{$}}
; GFX10: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_private_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_private_0(
    i32* %in, i32 addrspace(5)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(5)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_1:
; GFX89: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen glc slc{{$}}
; GFX10: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_private_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
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
; GFX10: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, off slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_global_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_global_0(
    i32* %in, i32 addrspace(1)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(1)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_global_1:
; GFX8:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
; GFX9:  global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX10: global_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}] slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_global_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
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
; GFX10:         .amdhsa_kernel nontemporal_local_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_local_0(
    i32* %in, i32 addrspace(3)* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32 addrspace(3)* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_local_1:
; GCN: ds_write_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel nontemporal_local_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
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
; GFX10: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_flat_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_flat_0(
    i32* %in, i32* %out) {
entry:
  %val = load i32, i32* %in, align 4
  store i32 %val, i32* %out, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}nontemporal_flat_1:
; GFX89: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc slc{{$}}
; GFX10: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_flat_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_flat_1(
    i32* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val = load i32, i32* %in, align 4
  %out.gep = getelementptr inbounds i32, i32* %out, i32 %tid
  store i32 %val, i32* %out.gep, !nontemporal !0
  ret void
}

; GCN-LABEL: {{^}}system_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}system_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}system_release:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out release, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel system_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") release, align 4
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel singlethread_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("singlethread") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_release:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") release, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel agent_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("agent") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") monotonic, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_release:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") release, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel workgroup_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("workgroup") seq_cst, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_unordered(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") unordered, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_monotonic(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") monotonic, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_release(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") release, align 4
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel wavefront_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst(
    i32 %in, i32* %out) {
entry:
  store atomic i32 %in, i32* %out syncscope("wavefront") seq_cst, align 4
  ret void
}

!0 = !{i32 1}
