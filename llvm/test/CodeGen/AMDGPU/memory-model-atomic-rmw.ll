; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @system_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in monotonic
  ret void
}

; CHECK-LABEL: {{^}}system_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in acquire
  ret void
}

; CHECK-LABEL: {{^}}system_release
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @system_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in release
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in acq_rel
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @system_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in seq_cst
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @agent_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(2) monotonic
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(2) acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @agent_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(2) release
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(2) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
define void @agent_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(2) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}work_group_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(3) monotonic
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(3) acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(3) release
  ret void
}

; CHECK-LABEL: {{^}}work_group_acq_rel
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(3) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @work_group_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(3) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(4) monotonic
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(4) acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(4) release
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(4) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @wavefront_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(4) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}image_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(5) monotonic
  ret void
}

; CHECK-LABEL: {{^}}image_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(5) acquire
  ret void
}

; CHECK-LABEL: {{^}}image_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(5) release
  ret void
}

; CHECK-LABEL: {{^}}image_acq_rel
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(5) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @image_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in synchscope(5) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_monotonic(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in singlethread monotonic
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acquire(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in singlethread acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_release(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in singlethread release
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acq_rel
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_acq_rel(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in singlethread acq_rel
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
define void @signal_handler_seq_cst(i32 addrspace(4)* %out, i32 %in) {
  %val = atomicrmw volatile xchg i32 addrspace(4)* %out, i32 %in singlethread seq_cst
  ret void
}
