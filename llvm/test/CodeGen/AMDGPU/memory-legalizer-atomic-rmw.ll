; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_monotonic(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acquire(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in acquire
  ret void
}

; CHECK-LABEL: {{^}}system_release
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @system_release(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in release
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_acq_rel(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in acq_rel
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @system_seq_cst(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; CHECK-LABEL: {{^}}singlethread_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_monotonic(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("singlethread") monotonic
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acquire(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("singlethread") acquire
  ret void
}

; CHECK-LABEL: {{^}}singlethread_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_release(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("singlethread") release
  ret void
}

; CHECK-LABEL: {{^}}singlethread_acq_rel
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_acq_rel(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("singlethread") acq_rel
  ret void
}

; CHECK-LABEL: {{^}}singlethread_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @singlethread_seq_cst(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("singlethread") seq_cst
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_monotonic(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("agent") monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acquire(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("agent") acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @agent_release(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("agent") release
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_acq_rel(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("agent") acq_rel
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK:       s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT:  s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT:  buffer_wbinvl1_vol
define amdgpu_kernel void @agent_seq_cst(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("agent") seq_cst
  ret void
}

; CHECK-LABEL: {{^}}workgroup_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_monotonic(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("workgroup") monotonic
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acquire(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("workgroup") acquire
  ret void
}

; CHECK-LABEL: {{^}}workgroup_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_release(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("workgroup") release
  ret void
}

; CHECK-LABEL: {{^}}workgroup_acq_rel
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_acq_rel(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("workgroup") acq_rel
  ret void
}

; CHECK-LABEL: {{^}}workgroup_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @workgroup_seq_cst(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("workgroup") seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_monotonic(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("wavefront") monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acquire(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("wavefront") acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_release(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("wavefront") release
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_acq_rel(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("wavefront") acq_rel
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT:   s_waitcnt vmcnt(0){{$}}
; CHECK-NOT:   buffer_wbinvl1_vol
define amdgpu_kernel void @wavefront_seq_cst(
    i32 addrspace(4)* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in syncscope("wavefront") seq_cst
  ret void
}
