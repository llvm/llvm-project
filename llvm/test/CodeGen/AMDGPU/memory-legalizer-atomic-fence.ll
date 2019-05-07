; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx600 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GFX6,GFX68 %s
; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GFX8,GFX68 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GFX8,GFX68 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3 -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GFX10,GFX10WGP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+code-object-v3,+cumode -verify-machineinstrs < %s | FileCheck -check-prefixes=FUNC,GCN,GFX10,GFX10CU %s

; FUNC-LABEL: {{^}}system_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0){{$}}
; GFX6-NEXT:  buffer_wbinvl1{{$}}
; GFX8:       s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol{{$}}
; GFX10:      s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire() {
entry:
  fence syncscope("one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release() {
entry:
  fence syncscope("one-as") release
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel() {
entry:
  fence syncscope("one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}system_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst() {
entry:
  fence syncscope("one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acquire() {
entry:
  fence syncscope("singlethread-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_release() {
entry:
  fence syncscope("singlethread-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acq_rel() {
entry:
  fence syncscope("singlethread-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}singlethread_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst() {
entry:
  fence syncscope("singlethread-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0){{$}}
; GFX6-NEXT:  buffer_wbinvl1{{$}}
; GFX8:       s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol{{$}}
; GFX10:      s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire() {
entry:
  fence syncscope("agent-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release() {
entry:
  fence syncscope("agent-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel() {
entry:
  fence syncscope("agent-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}agent_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst() {
entry:
  fence syncscope("agent-one-as") seq_cst
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_one_as_acquire:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire() {
entry:
  fence syncscope("workgroup-one-as") acquire
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_one_as_release:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NOT:     buffer_gl0_inv
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release() {
entry:
  fence syncscope("workgroup-one-as") release
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_one_as_acq_rel:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel() {
entry:
  fence syncscope("workgroup-one-as") acq_rel
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_one_as_seq_cst:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst() {
entry:
  fence syncscope("workgroup-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acquire() {
entry:
  fence syncscope("wavefront-one-as") acquire
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_release() {
entry:
  fence syncscope("wavefront-one-as") release
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acq_rel() {
entry:
  fence syncscope("wavefront-one-as") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}wavefront_one_as_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst() {
entry:
  fence syncscope("wavefront-one-as") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}system_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX6-NEXT:  buffer_wbinvl1{{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol{{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire() {
entry:
  fence acquire
  ret void
}

; FUNC-LABEL: {{^}}system_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release() {
entry:
  fence release
  ret void
}

; FUNC-LABEL: {{^}}system_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel() {
entry:
  fence acq_rel
  ret void
}

; FUNC-LABEL: {{^}}system_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel system_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst() {
entry:
  fence seq_cst
  ret void
}

; FUNC-LABEL: {{^}}singlethread_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acquire() {
entry:
  fence syncscope("singlethread") acquire
  ret void
}

; FUNC-LABEL: {{^}}singlethread_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_release() {
entry:
  fence syncscope("singlethread") release
  ret void
}

; FUNC-LABEL: {{^}}singlethread_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acq_rel() {
entry:
  fence syncscope("singlethread") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}singlethread_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel singlethread_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst() {
entry:
  fence syncscope("singlethread") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}agent_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX6-NEXT:  buffer_wbinvl1{{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol{{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire() {
entry:
  fence syncscope("agent") acquire
  ret void
}

; FUNC-LABEL: {{^}}agent_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release() {
entry:
  fence syncscope("agent") release
  ret void
}

; FUNC-LABEL: {{^}}agent_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel() {
entry:
  fence syncscope("agent") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}agent_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GFX6:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX6:       buffer_wbinvl1{{$}}
; GFX8:       buffer_wbinvl1_vol{{$}}
; GFX10-NEXT: buffer_gl0_inv{{$}}
; GFX10-NEXT: buffer_gl1_inv{{$}}
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel agent_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst() {
entry:
  fence syncscope("agent") seq_cst
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_acquire:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire() {
entry:
  fence syncscope("workgroup") acquire
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_release:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10-NOT:     buffer_gl0_inv
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release() {
entry:
  fence syncscope("workgroup") release
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_acq_rel:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel() {
entry:
  fence syncscope("workgroup") acq_rel
  ret void
}

; FUNC-LABEL:    {{^}}workgroup_seq_cst:
; GCN:           %bb.0
; GFX68-NOT:     s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv{{$}}
; GCN-NOT:       ATOMIC_FENCE
; GCN:           s_endpgm
; GFX10:         .amdhsa_kernel workgroup_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst() {
entry:
  fence syncscope("workgroup") seq_cst
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acquire:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acquire() {
entry:
  fence syncscope("wavefront") acquire
  ret void
}

; FUNC-LABEL: {{^}}wavefront_release:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_release() {
entry:
  fence syncscope("wavefront") release
  ret void
}

; FUNC-LABEL: {{^}}wavefront_acq_rel:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acq_rel() {
entry:
  fence syncscope("wavefront") acq_rel
  ret void
}

; FUNC-LABEL: {{^}}wavefront_seq_cst:
; GCN:        %bb.0
; GCN-NOT:    ATOMIC_FENCE
; GCN:        s_endpgm
; GFX10:         .amdhsa_kernel wavefront_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst() {
entry:
  fence syncscope("wavefront") seq_cst
  ret void
}
