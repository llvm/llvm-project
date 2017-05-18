; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=FUNC -check-prefix=GCN -check-prefix=VI %s

; FUNC-LABEL: {{^}}system_acquire
; GCN: BB#0
; SI:  s_waitcnt vmcnt(0){{$}}
; SI-NEXT:  buffer_wbinvl1{{$}}
; VI:  s_waitcnt vmcnt(0){{$}}
; VI-NEXT:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @system_acquire() {
  fence acquire
  ret void
}

; FUNC-LABEL: {{^}}system_release
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; GCN: s_endpgm
define amdgpu_kernel void @system_release() {
  fence release
  ret void
}

; FUNC-LABEL: {{^}}system_acq_rel
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; SI:  buffer_wbinvl1{{$}}
; VI:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @system_acq_rel() {
  fence acq_rel
  ret void
}

; FUNC-LABEL: {{^}}system_seq_cst
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; SI:  buffer_wbinvl1{{$}}
; VI:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @system_seq_cst() {
  fence seq_cst
  ret void
}

; FUNC-LABEL: {{^}}agent_acquire
; GCN: BB#0
; SI:  s_waitcnt vmcnt(0){{$}}
; SI-NEXT:  buffer_wbinvl1{{$}}
; VI:  s_waitcnt vmcnt(0){{$}}
; VI-NEXT:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @agent_acquire() {
  fence syncscope(2) acquire
  ret void
}

; FUNC-LABEL: {{^}}agent_release
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; GCN: s_endpgm
define amdgpu_kernel void @agent_release() {
  fence syncscope(2) release
  ret void
}

; FUNC-LABEL: {{^}}agent_acq_rel
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; SI:  buffer_wbinvl1{{$}}
; VI:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @agent_acq_rel() {
  fence syncscope(2) acq_rel
  ret void
}

; FUNC-LABEL: {{^}}agent_seq_cst
; GCN: BB#0
; GCN: s_waitcnt vmcnt(0){{$}}
; SI:  buffer_wbinvl1{{$}}
; VI:  buffer_wbinvl1_vol{{$}}
; GCN: s_endpgm
define amdgpu_kernel void @agent_seq_cst() {
  fence syncscope(2) seq_cst
  ret void
}

; FUNC-LABEL: {{^}}work_group_acquire
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @work_group_acquire() {
  fence syncscope(3) acquire
  ret void
}

; FUNC-LABEL: {{^}}work_group_release
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @work_group_release() {
  fence syncscope(3) release
  ret void
}

; FUNC-LABEL: {{^}}work_group_acq_rel
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @work_group_acq_rel() {
  fence syncscope(3) acq_rel
  ret void
}

; FUNC-LABEL: {{^}}work_group_seq_cst
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @work_group_seq_cst() {
  fence syncscope(3) seq_cst
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acquire
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @wavefront_acquire() {
  fence syncscope(4) acquire
  ret void
}

; FUNC-LABEL: {{^}}wavefront_release
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @wavefront_release() {
  fence syncscope(4) release
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acq_rel
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @wavefront_acq_rel() {
  fence syncscope(4) acq_rel
  ret void
}

; FUNC-LABEL: {{^}}wavefront_seq_cst
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @wavefront_seq_cst() {
  fence syncscope(4) seq_cst
  ret void
}

; FUNC-LABEL: {{^}}image_acquire
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @image_acquire() {
  fence syncscope(5) acquire
  ret void
}

; FUNC-LABEL: {{^}}image_release
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @image_release() {
  fence syncscope(5) release
  ret void
}

; FUNC-LABEL: {{^}}image_acq_rel
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @image_acq_rel() {
  fence syncscope(5) acq_rel
  ret void
}

; FUNC-LABEL: {{^}}image_seq_cst
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @image_seq_cst() {
  fence syncscope(5) seq_cst
  ret void
}

; FUNC-LABEL: {{^}}signal_handler_acquire
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @signal_handler_acquire() {
  fence singlethread acquire
  ret void
}

; FUNC-LABEL: {{^}}signal_handler_release
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @signal_handler_release() {
  fence singlethread release
  ret void
}

; FUNC-LABEL: {{^}}signal_handler_acq_rel
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @signal_handler_acq_rel() {
  fence singlethread acq_rel
  ret void
}

; FUNC-LABEL: {{^}}signal_handler_seq_cst
; GCN: BB#0
; GCN: s_endpgm
define amdgpu_kernel void @signal_handler_seq_cst() {
  fence singlethread seq_cst
  ret void
}
