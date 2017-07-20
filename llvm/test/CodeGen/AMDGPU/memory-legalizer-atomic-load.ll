; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @system_unordered(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @system_monotonic(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @system_acquire(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @system_seq_cst(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}singlethread_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @singlethread_unordered(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("singlethread") unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}singlethread_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @singlethread_monotonic(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("singlethread") monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @singlethread_acquire(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("singlethread") acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @singlethread_seq_cst(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("singlethread") seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @agent_unordered(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("agent") unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @agent_monotonic(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("agent") monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @agent_acquire(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("agent") acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @agent_seq_cst(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("agent") seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}workgroup_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @workgroup_unordered(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("workgroup") unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}workgroup_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @workgroup_monotonic(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("workgroup") monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @workgroup_acquire(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("workgroup") acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @workgroup_seq_cst(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("workgroup") seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @wavefront_unordered(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("wavefront") unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @wavefront_monotonic(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("wavefront") monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @wavefront_acquire(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("wavefront") acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
; CHECK:       flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define amdgpu_kernel void @wavefront_seq_cst(
    i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
entry:
  %val = load atomic i32, i32 addrspace(4)* %in syncscope("wavefront") seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}
