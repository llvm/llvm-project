; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @system_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @system_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @system_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @system_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @agent_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(2) unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @agent_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(2) monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @agent_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(2) acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}] glc{{$}}
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @agent_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(2) seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}work_group_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @work_group_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(3) unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}work_group_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @work_group_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(3) monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @work_group_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(3) acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @work_group_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(3) seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @wavefront_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(4) unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @wavefront_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(4) monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @wavefront_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(4) acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @wavefront_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(4) seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}image_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @image_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(5) unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}image_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @image_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(5) monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}image_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @image_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(5) acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @image_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in syncscope(5) seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_unordered
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @signal_handler_unordered(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in singlethread unordered, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_monotonic
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @signal_handler_monotonic(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in singlethread monotonic, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acquire
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @signal_handler_acquire(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in singlethread acquire, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK: flat_load_dword [[RET:v[0-9]+]], v[{{[0-9]+}}:{{[0-9]+}}]{{$}}
; CHECK-NOT: s_waitcnt vmcnt(0){{$}}
; CHECK-NOT: buffer_wbinvl1_vol
; CHECK: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RET]]
define void @signal_handler_seq_cst(i32 addrspace(4)* %in, i32 addrspace(4)* %out) {
  %val = load atomic i32, i32 addrspace(4)* %in singlethread seq_cst, align 4
  store i32 %val, i32 addrspace(4)* %out
  ret void
}
