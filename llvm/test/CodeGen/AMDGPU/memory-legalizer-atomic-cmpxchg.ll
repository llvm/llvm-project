; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s

; GCN-LABEL: {{^}}system_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}system_release_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @system_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in release monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire acquire
  ret void
}

; GCN-LABEL: {{^}}system_release_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in release acquire
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") release monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") release acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_release_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") release monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}agent_release_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") release acquire
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_release_monotonic:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") release monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_acq_rel_monotonic:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_seq_cst_monotonic:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}workgroup_release_acquire:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") release acquire
  ret void
}

; GCN-LABEL: {{^}}workgroup_acq_rel_acquire:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}workgroup_seq_cst_acquire:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}workgroup_seq_cst_seq_cst:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") release monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") release acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst seq_cst
  ret void
}
