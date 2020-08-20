; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX8,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3 -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10WGP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3,+cumode -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,GFX10,GFX10CU %s

declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}system_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("one-as") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("one-as") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("one-as") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX89-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("one-as") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread-one-as") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread-one-as") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread-one-as") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread-one-as") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent-one-as") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent-one-as") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent-one-as") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX89-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent-one-as") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}workgroup_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup-one-as") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:    {{^}}workgroup_one_as_monotonic:
; GCN-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:        flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU-NOT:  flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT:    buffer_wbinvl1_vol
; GFX10-NOT:    buffer_gl{{[01]}}_inv
; GCN:          flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup-one-as") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:     s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:         flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU-NOT:   flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NOT:     buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GCN:           flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup-one-as") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0
; GFX89:         flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX89-NOT:     buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GCN:           flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup-one-as") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_one_as_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront-one-as") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront-one-as") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront-one-as") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront-one-as") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_0:
; GFX89: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen glc slc{{$}}
; GFX10: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_private_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_private_0(
    i32 addrspace(5)* %in, i32* %out) {
entry:
  %val = load i32, i32 addrspace(5)* %in, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_private_1:
; GFX89: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen glc slc{{$}}
; GFX10: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0 offen slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_private_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_private_1(
    i32 addrspace(5)* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(5)* %in, i32 %tid
  %val = load i32, i32 addrspace(5)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_global_0:
; GCN: s_load_dword s{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 0x0{{$}}
; GFX10:         .amdhsa_kernel nontemporal_global_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_global_0(
    i32 addrspace(1)* %in, i32* %out) {
entry:
  %val = load i32, i32 addrspace(1)* %in, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_global_1:
; GFX8:  flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX9:  global_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX10: global_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}] slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_global_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_global_1(
    i32 addrspace(1)* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_local_0:
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel nontemporal_local_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_local_0(
    i32 addrspace(3)* %in, i32* %out) {
entry:
  %val = load i32, i32 addrspace(3)* %in, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_local_1:
; GCN: ds_read_b32 v{{[0-9]+}}, v{{[0-9]+}}{{$}}
; GFX10:         .amdhsa_kernel nontemporal_local_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_local_1(
    i32 addrspace(3)* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32 addrspace(3)* %in, i32 %tid
  %val = load i32, i32 addrspace(3)* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_flat_0:
; GFX89: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_flat_0
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_flat_0(
    i32* %in, i32* %out) {
entry:
  %val = load i32, i32* %in, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}nontemporal_flat_1:
; GFX89: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] glc slc{{$}}
; GFX10: flat_load_dword v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}] slc{{$}}
; GFX10:         .amdhsa_kernel nontemporal_flat_1
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @nontemporal_flat_1(
    i32* %in, i32* %out) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %val.gep = getelementptr inbounds i32, i32* %in, i32 %tid
  %val = load i32, i32* %val.gep, align 4, !nontemporal !0
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GFX89-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX89-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GFX89-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel system_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel singlethread_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("singlethread") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GFX89-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst:
; GFX89:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX89-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc dlc{{$}}
; GFX89-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NEXT: buffer_wbinvl1_vol
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GCN:        flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel agent_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("agent") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}workgroup_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:    {{^}}workgroup_monotonic:
; GCN-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:        flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:     flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU-NOT:  flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GCN-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT:    buffer_wbinvl1_vol
; GFX10-NOT:    buffer_gl{{[01]}}_inv
; GCN:          flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire:
; GFX10-NOT:     s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89:         flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU-NOT:   flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX89:         s_waitcnt lgkmcnt(0){{$}}
; GFX89:         s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NOT:     buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GCN:           flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst:
; GFX89-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0
; GFX89:         flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX10WGP:      flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; GFX10CU:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GFX89:         s_waitcnt lgkmcnt(0){{$}}
; GFX89:         s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX89-NOT:     buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GCN:           flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel workgroup_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("workgroup") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_unordered:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_unordered
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_unordered(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront") unordered, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_monotonic(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront") monotonic, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acquire(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront") acquire, align 4
  store i32 %val, i32* %out
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX89-NOT: buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GCN:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
; GFX10:         .amdhsa_kernel wavefront_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst(
    i32* %in, i32* %out) {
entry:
  %val = load atomic i32, i32* %in syncscope("wavefront") seq_cst, align 4
  store i32 %val, i32* %out
  ret void
}

!0 = !{i32 1}
