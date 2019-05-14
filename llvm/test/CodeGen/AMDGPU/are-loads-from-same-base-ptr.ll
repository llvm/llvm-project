; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; TII::areLoadsFromSameBasePtr failed because the offset for atomics
; is different from a normal load due to the data operand.

; GCN-LABEL: {{^}}are_loads_from_same_base_ptr_ds_atomic:
; GCN: global_load_dword
; GCN: ds_min_u32
; GCN: ds_max_u32
define amdgpu_kernel void @are_loads_from_same_base_ptr_ds_atomic(i32 addrspace(1)* %arg0, i32 addrspace(3)* noalias %ptr0) #0 {
  %tmp1 = load volatile i32, i32 addrspace(1)* %arg0
  %tmp2 = atomicrmw umin i32 addrspace(3)* %ptr0, i32 %tmp1 seq_cst
  %tmp3 = atomicrmw umax i32 addrspace(3)* %ptr0, i32 %tmp1 seq_cst
  ret void
}

attributes #0 = { nounwind }
