; RUN: llc -mtriple=amdgcn-amd- -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx803 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX8 %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX10WGP %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1010 -mattr=+cumode -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10,GFX10CU %s

; GCN-LABEL: {{^}}system_one_as_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel system_one_as_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_release_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:   buffer_wbinvl1_vol
; GFX10-NOT:  buffer_gl._inv
; GFX10:         .amdhsa_kernel system_one_as_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") release monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}system_one_as_release_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") release acquire
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_one_as_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_one_as_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; Gfx8-NOT:  buffer_wbinvl1_vol
; GCN-NOT:   buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_one_as_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") release monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") release acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_one_as_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_one_as_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_one_as_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread-one-as") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel agent_one_as_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_release_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_one_as_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") release monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_monotonic:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_release_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") release acquire
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_acquire:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_seq_cst:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_one_as_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel workgroup_one_as_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") monotonic monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire_monotonic:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:     s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acquire monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_release_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:       buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_one_as_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") release monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acq_rel monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire_acquire:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acquire acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_release_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") release acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acq_rel acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_seq_cst:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") release monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") release acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_one_as_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_one_as_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_one_as_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront-one-as") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire_monotonic_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel_monotonic_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_monotonic_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acquire_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_release_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_acq_rel_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_one_as_seq_cst_seq_cst_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_one_as_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_one_as_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("one-as") seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire_monotonic_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel_monotonic_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_monotonic_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acquire_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_release_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_acq_rel_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_acquire_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_one_as_seq_cst_seq_cst_ret:
; GCN:        s_waitcnt vmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GCN-NEXT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_one_as_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_one_as_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent-one-as") seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire_monotonic_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel_monotonic_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_monotonic_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acquire_acquire_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_one_as_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_release_acquire_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_acq_rel_acquire_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_acquire_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_one_as_seq_cst_seq_cst_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_one_as_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_one_as_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup-one-as") seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GCN-NOT:   s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel system_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}system_release_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:   buffer_wbinvl1_vol
; GFX10-NOT:  buffer_gl._inv
; GFX10:         .amdhsa_kernel system_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in release monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}system_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:  s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire acquire
  ret void
}

; GCN-LABEL: {{^}}system_release_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in release acquire
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_seq_cst:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}singlethread_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; Gfx8-NOT:  buffer_wbinvl1_vol
; GCN-NOT:   buffer_gl{{[01]}}_inv
; GFX10:         .amdhsa_kernel singlethread_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") release monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}singlethread_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") release acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}singlethread_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel singlethread_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @singlethread_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("singlethread") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}agent_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel agent_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_monotonic:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_release_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:    buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel agent_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") release monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_monotonic:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_acquire:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}agent_release_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") release acquire
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_acquire:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_seq_cst:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}workgroup_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel workgroup_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") monotonic monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire_monotonic:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GFX10-NOT:     s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_release_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN-NOT:       buffer_{{wbinvl1_vol|gl._inv}}
; GFX10:         .amdhsa_kernel workgroup_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") release monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_monotonic:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst monotonic
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire_acquire:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_release_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") release acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_acquire:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst acquire
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_seq_cst:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}wavefront_monotonic_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_monotonic_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_monotonic_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") monotonic monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_acquire_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acquire_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acquire monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_release_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_release_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_release_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") release monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_acq_rel_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acq_rel_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_monotonic:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_seq_cst_monotonic
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst_monotonic(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst monotonic
  ret void
}

; GCN-LABEL: {{^}}wavefront_acquire_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_acquire_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acquire_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acquire acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_release_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_release_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_release_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") release acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_acq_rel_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_acq_rel_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_acq_rel_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") acq_rel acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_acquire:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_seq_cst_acquire
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst_acquire(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst acquire
  ret void
}

; GCN-LABEL: {{^}}wavefront_seq_cst_seq_cst:
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:       flat_atomic_cmpswap v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}}{{$}}
; GCN-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10-NOT: s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GFX8-NOT:  buffer_wbinvl1_vol
; GFX10-NOT: buffer_gl{{[01]}}._inv
; GFX10:         .amdhsa_kernel wavefront_seq_cst_seq_cst
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @wavefront_seq_cst_seq_cst(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("wavefront") seq_cst seq_cst
  ret void
}

; GCN-LABEL: {{^}}system_acquire_monotonic_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_monotonic_ret:
; GCN:        s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_monotonic_ret:
; GCN:        s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_acquire_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_release_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_acq_rel_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}system_seq_cst_seq_cst_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel system_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @system_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_monotonic_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_monotonic_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_monotonic_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acquire_acquire_ret:
; GCN-NOT:    s_waitcnt vmcnt(0){{$}}
; GCN-NOT:    s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:        flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_release_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_acq_rel_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_acquire_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}agent_seq_cst_seq_cst_ret:
; GFX8:       s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10:      s_waitcnt lgkmcnt(0){{$}}
; GFX10:      s_waitcnt_vscnt null, 0x0{{$}}
; GCN-NEXT:   flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8-NEXT:  s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX8-NEXT:  buffer_wbinvl1_vol
; GFX10-NEXT: buffer_gl0_inv
; GFX10-NEXT: buffer_gl1_inv
; GFX10:         .amdhsa_kernel agent_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @agent_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("agent") seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire_monotonic_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:          s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_acquire_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel_monotonic_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:          s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_monotonic_ret:
; GFX8-NOT:      s_waitcnt vmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:          s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_monotonic_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_monotonic_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst monotonic
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acquire_acquire_ret:
; GCN-NOT:       s_waitcnt vmcnt(0){{$}}
; GCN-NOT:       s_waitcnt_v{{[ms]}}cnt {{[^,]+, (0x)*0$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:          s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   buffer_gl0_inv
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10:         .amdhsa_kernel workgroup_acquire_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acquire_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acquire acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_release_acquire_ret:
; GFX8:          s_waitcnt lgkmcnt(0){{$}}
; GFX8:          flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GFX10:         flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX8:          s_waitcnt vmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_release_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_release_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") release acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_acq_rel_acquire_ret:
; GFX8:          s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_acq_rel_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_acq_rel_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") acq_rel acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_acquire_ret:
; GFX8:          s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP-NEXT: buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_acquire_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_acquire_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst acquire
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}

; GCN-LABEL:     {{^}}workgroup_seq_cst_seq_cst_ret:
; GFX8:          s_waitcnt lgkmcnt(0){{$}}
; GFX10WGP:      s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10WGP-NEXT: s_waitcnt_vscnt null, 0x0{{$}}
; GFX10CU-NOT:   s_waitcnt vmcnt(0){{$}}
; GFX10CU-NOT:   s_waitcnt_vscnt null, 0x0{{$}}
; GCN:           flat_atomic_cmpswap v{{[0-9]+}}, v[{{[0-9]+\:[0-9]+}}], v[{{[0-9]+}}:{{[0-9]+}}]{{( offset:[0-9]+)*}} glc{{$}}
; GFX10WGP-NEXT: s_waitcnt vmcnt(0) lgkmcnt(0){{$}}
; GFX10CU:       s_waitcnt vmcnt(0){{$}}
; GFX8-NOT:      buffer_wbinvl1_vol
; GFX10WGP:      buffer_gl0_inv
; GFX10CU-NOT:   buffer_gl0_inv
; GFX10:         .amdhsa_kernel workgroup_seq_cst_seq_cst_ret
; GFX10WGP-NOT:  .amdhsa_workgroup_processor_mode 0
; GFX10CU:       .amdhsa_workgroup_processor_mode 0
; GFX10-NOT:     .amdhsa_memory_ordered 0
define amdgpu_kernel void @workgroup_seq_cst_seq_cst_ret(
    i32* %out, i32 %in, i32 %old) {
entry:
  %gep = getelementptr i32, i32* %out, i32 4
  %val = cmpxchg volatile i32* %gep, i32 %old, i32 %in syncscope("workgroup") seq_cst seq_cst
  %val0 = extractvalue { i32, i1 } %val, 0
  store i32 %val0, i32* %out, align 4
  ret void
}
