; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @system_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @system_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_release
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @system_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out release, align 4
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @system_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @agent_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(2) unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @agent_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(2) monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @agent_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(2) release, align 4
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @agent_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(2) seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}work_group_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @work_group_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(3) unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}work_group_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @work_group_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(3) monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}work_group_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @work_group_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(3) release, align 4
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @work_group_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(3) seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @wavefront_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(4) unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @wavefront_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(4) monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @wavefront_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(4) release, align 4
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @wavefront_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(4) seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}image_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @image_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(5) unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}image_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @image_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(5) monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}image_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @image_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(5) release, align 4
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @image_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out synchscope(5) seq_cst, align 4
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @signal_handler_unordered(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out singlethread unordered, align 4
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @signal_handler_monotonic(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out singlethread monotonic, align 4
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_release
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @signal_handler_release(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out singlethread release, align 4
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_store_dword v[{{[0-9]+}}:{{[0-9]+}}], {{v[0-9]+}}{{$}}
define void @signal_handler_seq_cst(i32 %in, i32 addrspace(4)* %out) {
  store atomic i32 %in, i32 addrspace(4)* %out singlethread seq_cst, align 4
  ret void
}
