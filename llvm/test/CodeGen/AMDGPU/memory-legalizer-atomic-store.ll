; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_unordered(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_monotonic(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_release
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_release(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out release, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @system_seq_cst(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}singlethread_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_unordered(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("singlethread") unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}singlethread_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_monotonic(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("singlethread") monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}singlethread_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_release(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("singlethread") release, align 4
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @singlethread_seq_cst(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("singlethread") seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_unordered(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("agent") unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_monotonic(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("agent") monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_release(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("agent") release, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @agent_seq_cst(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("agent") seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}workgroup_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_unordered(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("workgroup") unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}workgroup_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_monotonic(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("workgroup") monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}workgroup_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_release(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("workgroup") release, align 4
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @workgroup_seq_cst(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("workgroup") seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_unordered
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_unordered(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("wavefront") unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_monotonic(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("wavefront") monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_release(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("wavefront") release, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define amdgpu_kernel void @wavefront_seq_cst(
    i32 %in, i32 addrspace(4)* %out) {
entry:
  store atomic i32 %in, i32 addrspace(4)* %out syncscope("wavefront") seq_cst, align 4
  ret void
}
