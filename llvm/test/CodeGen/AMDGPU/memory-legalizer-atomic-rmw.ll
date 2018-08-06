; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s

; GCN-LABEL: {{^}}system_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acquire
  ret void
}

; GCN-LABEL: {{^}}system_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @system_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in release
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel:
; GCN:         s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:    flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:    s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acq_rel
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") release
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") acq_rel
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acquire
  ret void
}

; GCN-LABEL: {{^}}agent_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") release
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acq_rel
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") monotonic
  ret void
}

; GCN-LABEL: {{^}}workgroup_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acquire
  ret void
}

; GCN-LABEL: {{^}}workgroup_release:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") release
  ret void
}

; GCN-LABEL: {{^}}workgroup_acq_rel:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acq_rel
  ret void
}

; GCN-LABEL: {{^}}workgroup_seq_cst:
; GFX8-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") release
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") acq_rel
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") seq_cst
  ret void
}
