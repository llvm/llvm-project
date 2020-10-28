; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX10WGP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+cumode -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX10CU %s

; GCN-LABEL: {{^}}system_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel system_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") acquire
  ret void
}

; GCN-LABEL: {{^}}system_one_as_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel system_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") release
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel:
; GCN:         s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT:  s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:    flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT:  s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") acq_rel
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread-one-as") monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread-one-as") acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread-one-as") release
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread-one-as") acq_rel
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread-one-as") seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") acquire
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_release:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") release
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") acq_rel
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_release:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:       buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") release
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") acq_rel
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_one_as_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront-one-as") monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_one_as_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront-one-as") acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_one_as_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront-one-as") release
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_one_as_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront-one-as") acq_rel
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront-one-as") seq_cst
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel_ret:
; GCN:         s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT:  s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:    flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:    s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("one-as") seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent-one-as") seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup-one-as") seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel system_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acquire
  ret void
}

; GCN-LABEL: {{^}}system_release:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel system_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in release
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:    flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acq_rel
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") release
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") acq_rel
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel singlethread_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("singlethread") seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acquire
  ret void
}

; GCN-LABEL: {{^}}agent_release:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") release
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acq_rel
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_release:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:       buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") release
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acq_rel
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_monotonic(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acquire(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_release:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_release
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_release(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") release
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_acq_rel
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acq_rel(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") acq_rel
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_swap v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:   buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel wavefront_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("wavefront") seq_cst
  ret void
}

; GCN-LABEL: {{^}}system_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_ret:
; GFX8:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:    flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_ret:
; GFX8:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_ret:
; GFX8:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_ret:
; GFX8:        s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:       s_waitcnt lgkmcnt(0){{$}}
; GFX10:       s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NEXT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:   buffer_wbinvl1_vol
; GFX10-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT:  buffer_gl0_inv
; GFX10-NEXT:  buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("agent") seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acquire
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") acq_rel
  store i32 %val, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_swap v{{[0-9]+}}, v[{{[0-9]+}}:{{[0-9]+}}], v{{[0-9]+}} glc{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_ret(
    i32* %out, i32 %in) {
entry:
  %val = atomicrmw volatile xchg i32* %out, i32 %in syncscope("workgroup") seq_cst
  store i32 %val, i32* %out, align 4
  ret void
}
