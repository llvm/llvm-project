; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @system_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_release_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @system_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in release monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}system_release_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in release acquire
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @agent_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_release_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @agent_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) release monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_monotonic
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) release acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_acquire
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(2) seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}work_group_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_release_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) release monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_acq_rel_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_release_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) release acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_acq_rel_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(3) seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) release monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) release acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(4) seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}image_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_release_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) release monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_acq_rel_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}image_release_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) release acquire
  ret void
}

; CHECK-LABEL: {{^}}image_acq_rel_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in syncscope(5) seq_cst seq_cst
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_monotonic_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_monotonic_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread monotonic monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acquire_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acquire_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread acquire monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_release_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_release_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread release monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acq_rel_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acq_rel_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread acq_rel monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_seq_cst_monotonic(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread seq_cst monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acquire_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acquire_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread acquire acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_release_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_release_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread release acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acq_rel_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acq_rel_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread acq_rel acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_seq_cst_acquire(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread seq_cst acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_seq_cst_seq_cst(i32 addrspace(4)* %out, i32 %in, i32 %old) {
  %gep = getelementptr i32, i32 addrspace(4)* %out, i32 4
  %val = cmpxchg volatile i32 addrspace(4)* %gep, i32 %old, i32 %in singlethread seq_cst seq_cst
  ret void
}
