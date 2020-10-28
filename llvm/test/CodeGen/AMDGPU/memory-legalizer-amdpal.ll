; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN9,CACHE_INV %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx700 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN9,CACHE_INV %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN9,CACHE_INV %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN9,CACHE_INV %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN10,CACHE_INV10 %s

; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx900 -amdgcn-skip-cache-invalidations -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN9,SKIP_CACHE_INV %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=gfx1010 -amdgcn-skip-cache-invalidations -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GCN10,SKIP_CACHE_INV %s


; FUNC-LABEL: {{^}}system_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_acquire() {
entry:
  fence acquire
  ret void
}

; FUNC-LABEL: {{^}}system_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN:        s_endpgm
define amdgpu_kernel void @system_release() {
entry:
  fence release
  ret void
}

; FUNC-LABEL: {{^}}system_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_acq_rel() {
entry:
  fence acq_rel
  ret void
}

; FUNC-LABEL: {{^}}system_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_seq_cst() {
entry:
  fence seq_cst
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_one_as_acquire() {
entry:
  fence syncscope("one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN:        s_endpgm
define amdgpu_kernel void @system_one_as_release() {
entry:
  fence syncscope("one-as") release
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_one_as_acq_rel() {
entry:
  fence syncscope("one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:    buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @system_one_as_seq_cst() {
entry:
  fence syncscope("one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}singlethread_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_acquire() {
entry:
  fence syncscope("singlethread") acquire
  ret void
}

; FUNC-LABEL: {{^}}singlethread_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_release() {
entry:
  fence syncscope("singlethread") release
  ret void
}

; FUNC-LABEL: {{^}}singlethread_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_acq_rel() {
entry:
  fence syncscope("singlethread") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}singlethread_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_seq_cst() {
entry:
  fence syncscope("singlethread") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_one_as_acquire() {
entry:
  fence syncscope("singlethread-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_one_as_release() {
entry:
  fence syncscope("singlethread-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_one_as_acq_rel() {
entry:
  fence syncscope("singlethread-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @singlethread_one_as_seq_cst() {
entry:
  fence syncscope("singlethread-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}agent_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_acquire() {
entry:
  fence syncscope("agent") acquire
  ret void
}

; FUNC-LABEL: {{^}}agent_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN:        s_endpgm
define amdgpu_kernel void @agent_release() {
entry:
  fence syncscope("agent") release
  ret void
}

; FUNC-LABEL: {{^}}agent_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_acq_rel() {
entry:
  fence syncscope("agent") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}agent_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_seq_cst() {
entry:
  fence syncscope("agent") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_one_as_acquire() {
entry:
  fence syncscope("agent-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN:        s_endpgm
define amdgpu_kernel void @agent_one_as_release() {
entry:
  fence syncscope("agent-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_one_as_acq_rel() {
entry:
  fence syncscope("agent-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; CACHE_INV:  buffer_wbinvl1{{$}}
; CACHE_INV10: buffer_gl0_inv
; CACHE_INV10: buffer_gl1_inv
; SKIP_CACHE_INV-NOT: buffer_wbinvl1{{$}}
; SKIP_CACHE_INV-NOT: buffer_gl
; GCN:        s_endpgm
define amdgpu_kernel void @agent_one_as_seq_cst() {
entry:
  fence syncscope("agent-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}workgroup_acquire:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_acquire() {
entry:
  fence syncscope("workgroup") acquire
  ret void
}

; FUNC-LABEL: {{^}}workgroup_release:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_release() {
entry:
  fence syncscope("workgroup") release
  ret void
}

; FUNC-LABEL: {{^}}workgroup_acq_rel:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_acq_rel() {
entry:
  fence syncscope("workgroup") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}workgroup_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_seq_cst() {
entry:
  fence syncscope("workgroup") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}workgroup_one_as_acquire:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0)
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_one_as_acquire() {
entry:
  fence syncscope("workgroup-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}workgroup_one_as_release:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0)
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_one_as_release() {
entry:
  fence syncscope("workgroup-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}workgroup_one_as_acq_rel:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0)
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_one_as_acq_rel() {
entry:
  fence syncscope("workgroup-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}workgroup_one_as_seq_cst:
; GCN:        %bb.0
; GCN9-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GCN10:      s_waitcnt vmcnt(0)
; GCN10:      s_waitcnt_vscnt null, 0x0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @workgroup_one_as_seq_cst() {
entry:
  fence syncscope("workgroup-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_acquire() {
entry:
  fence syncscope("wavefront") acquire
  ret void
}

; FUNC-LABEL: {{^}}wavefront_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_release() {
entry:
  fence syncscope("wavefront") release
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_acq_rel() {
entry:
  fence syncscope("wavefront") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}wavefront_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_seq_cst() {
entry:
  fence syncscope("wavefront") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_one_as_acquire() {
entry:
  fence syncscope("wavefront-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_one_as_release() {
entry:
  fence syncscope("wavefront-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_one_as_acq_rel() {
entry:
  fence syncscope("wavefront-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
define amdgpu_kernel void @wavefront_one_as_seq_cst() {
entry:
  fence syncscope("wavefront-one-as") seq_cst
  ret void
}
