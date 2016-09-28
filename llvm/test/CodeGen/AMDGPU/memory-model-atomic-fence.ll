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
  fence syncscope(2) acquire
  ret void
}

; CHECK-LABEL: {{^}}agent_release
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: s_endpgm
define void @agent_release() {
  fence syncscope(2) release
  ret void
}

; CHECK-LABEL: {{^}}agent_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @agent_acq_rel() {
  fence syncscope(2) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}agent_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_waitcnt vmcnt(0){{$}}
; CHECK-NEXT: buffer_wbinvl1_vol
; CHECK-NEXT: s_endpgm
define void @agent_seq_cst() {
  fence syncscope(2) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}work_group_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_acquire() {
  fence syncscope(3) acquire
  ret void
}

; CHECK-LABEL: {{^}}work_group_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_release() {
  fence syncscope(3) release
  ret void
}

; CHECK-LABEL: {{^}}work_group_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_acq_rel() {
  fence syncscope(3) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}work_group_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @work_group_seq_cst() {
  fence syncscope(3) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_acquire() {
  fence syncscope(4) acquire
  ret void
}

; CHECK-LABEL: {{^}}wavefront_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_release() {
  fence syncscope(4) release
  ret void
}

; CHECK-LABEL: {{^}}wavefront_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_acq_rel() {
  fence syncscope(4) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}wavefront_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @wavefront_seq_cst() {
  fence syncscope(4) seq_cst
  ret void
}

; CHECK-LABEL: {{^}}image_acquire
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_acquire() {
  fence syncscope(5) acquire
  ret void
}

; CHECK-LABEL: {{^}}image_release
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_release() {
  fence syncscope(5) release
  ret void
}

; CHECK-LABEL: {{^}}image_acq_rel
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_acq_rel() {
  fence syncscope(5) acq_rel
  ret void
}

; CHECK-LABEL: {{^}}image_seq_cst
; CHECK: BB#0
; CHECK-NEXT: s_endpgm
define void @image_seq_cst() {
  fence syncscope(5) seq_cst
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
