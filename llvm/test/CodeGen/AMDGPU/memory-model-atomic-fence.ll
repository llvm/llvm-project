; RUN: llc -mtriple=amdgcn-- -mcpu=fiji -verify-machineinstrs < %s | FileCheck %s

; CHECK-LABEL: {{^}}system_acquire
; CHECK: BB#0
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @system_acquire() {
  fence acquire
  ret void
}

; CHECK-LABEL: {{^}}system_release
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: s_endpgm
define void @system_release() {
  fence release
  ret void
}

; CHECK-LABEL: {{^}}system_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @system_acq_rel() {
  fence acq_rel
  ret void
}

; CHECK-LABEL: {{^}}system_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @system_seq_cst() {
  fence seq_cst
  ret void
}

; CHECK-LABEL: {{^}}agent_acquire
; CHECK: BB#0
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @agent_acquire() {
  fence synchscope(2) acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: s_endpgm
define void @agent_release() {
  fence synchscope(2) release
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @agent_acq_rel() {
  fence synchscope(2) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @agent_seq_cst() {
  fence synchscope(2) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_acquire() {
  fence synchscope(3) acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_release() {
  fence synchscope(3) release
  ret void
}

; CHECK-LABEL: {{^}}work_group_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_acq_rel() {
  fence synchscope(3) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_seq_cst() {
  fence synchscope(3) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_acquire() {
  fence synchscope(4) acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_release() {
  fence synchscope(4) release
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_acq_rel() {
  fence synchscope(4) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_seq_cst() {
  fence synchscope(4) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}image_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_acquire() {
  fence synchscope(5) acquire
  ret void
}

; CHECK-LABEL: {{^}}image_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_release() {
  fence synchscope(5) release
  ret void
}

; CHECK-LABEL: {{^}}image_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_acq_rel() {
  fence synchscope(5) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_seq_cst() {
  fence synchscope(5) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @signal_handler_acquire() {
  fence singlethread acquire
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @signal_handler_release() {
  fence singlethread release
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @signal_handler_acq_rel() {
  fence singlethread acq_rel
  ret void
}

; CHECK-LABEL: {{^}}signal_handler_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @signal_handler_seq_cst() {
  fence singlethread seq_cst
  ret void
}
