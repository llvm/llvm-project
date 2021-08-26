; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Combine on select c, (load x), (load y) -> load (select c, x, y)
; drops MachinePointerInfo, so it can't be relied on for correctness.

; GCN-LABEL: {{^}}select_ptr_crash_i64_flat:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2

; GCN: v_cmp_eq_u32
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32

; GCN-NOT: load_dword
; GCN: flat_load_dwordx2
; GCN-NOT: load_dword

; GCN: flat_store_dwordx2
define amdgpu_kernel void @select_ptr_crash_i64_flat(i32 %tmp, [8 x i32], i64* %ptr0, [8 x i32], i64* %ptr1, [8 x i32], i64 addrspace(1)* %ptr2) {
  %tmp2 = icmp eq i32 %tmp, 0
  %tmp3 = load i64, i64* %ptr0, align 8
  %tmp4 = load i64, i64* %ptr1, align 8
  %tmp5 = select i1 %tmp2, i64 %tmp3, i64 %tmp4
  store i64 %tmp5, i64 addrspace(1)* %ptr2, align 8
  ret void
}

; The transform currently doesn't happen for non-addrspace 0, but it
; should.

; GCN-LABEL: {{^}}select_ptr_crash_i64_global:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x0{{$}}
; GCN: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x0{{$}}
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32
; GCN: flat_store_dwordx2
define amdgpu_kernel void @select_ptr_crash_i64_global(i32 %tmp, [8 x i32], i64 addrspace(1)* %ptr0, [8 x i32], i64 addrspace(1)* %ptr1, [8 x i32], i64 addrspace(1)* %ptr2) {
  %tmp2 = icmp eq i32 %tmp, 0
  %tmp3 = load i64, i64 addrspace(1)* %ptr0, align 8
  %tmp4 = load i64, i64 addrspace(1)* %ptr1, align 8
  %tmp5 = select i1 %tmp2, i64 %tmp3, i64 %tmp4
  store i64 %tmp5, i64 addrspace(1)* %ptr2, align 8
  ret void
}

; GCN-LABEL: {{^}}select_ptr_crash_i64_local:
; GCN: ds_read_b64
; GCN: ds_read_b64
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32
; GCN: flat_store_dwordx2
define amdgpu_kernel void @select_ptr_crash_i64_local(i32 %tmp, i64 addrspace(3)* %ptr0, i64 addrspace(3)* %ptr1, i64 addrspace(1)* %ptr2) {
  %tmp2 = icmp eq i32 %tmp, 0
  %tmp3 = load i64, i64 addrspace(3)* %ptr0, align 8
  %tmp4 = load i64, i64 addrspace(3)* %ptr1, align 8
  %tmp5 = select i1 %tmp2, i64 %tmp3, i64 %tmp4
  store i64 %tmp5, i64 addrspace(1)* %ptr2, align 8
  ret void
}

; The transform will break addressing mode matching, so unclear it
; would be good to do

; GCN-LABEL: {{^}}select_ptr_crash_i64_local_offsets:
; GCN: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:128
; GCN: ds_read_b64 {{v\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}} offset:512
; GCN: v_cndmask_b32
; GCN: v_cndmask_b32
define amdgpu_kernel void @select_ptr_crash_i64_local_offsets(i32 %tmp, i64 addrspace(3)* %ptr0, i64 addrspace(3)* %ptr1, i64 addrspace(1)* %ptr2) {
  %tmp2 = icmp eq i32 %tmp, 0
  %gep0 = getelementptr inbounds i64, i64 addrspace(3)* %ptr0, i64 16
  %gep1 = getelementptr inbounds i64, i64 addrspace(3)* %ptr1, i64 64
  %tmp3 = load i64, i64 addrspace(3)* %gep0, align 8
  %tmp4 = load i64, i64 addrspace(3)* %gep1, align 8
  %tmp5 = select i1 %tmp2, i64 %tmp3, i64 %tmp4
  store i64 %tmp5, i64 addrspace(1)* %ptr2, align 8
  ret void
}
