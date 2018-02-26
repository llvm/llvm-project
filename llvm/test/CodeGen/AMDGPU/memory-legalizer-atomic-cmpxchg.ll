; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_monotonic_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_monotonic_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_release_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_release_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in release monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}system_release_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_release_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in release acquire
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_seq_cst(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}singlethread_monotonic_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_monotonic_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acquire_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_release_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") release monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acq_rel_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acquire_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}singlethread_release_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") release acquire
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acq_rel_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_seq_cst(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_monotonic_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_release_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") release monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_monotonic
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") release acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_acquire
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_seq_cst(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}workgroup_monotonic_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_monotonic_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acquire_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_release_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") release monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acq_rel_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acquire_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}workgroup_release_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") release acquire
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acq_rel_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_seq_cst(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_monotonic_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") release monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_monotonic(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") release acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_acquire(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_seq_cst(
    i32 addrspace(4)* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst seq_cst
  ret void
}
