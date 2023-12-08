; RUN: llc -march=amdgcn -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,SI,SICIVI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,SICIVI,GFX89 %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -amdgpu-atomic-optimizer-strategy=None -verify-machineinstrs < %s | FileCheck -enable-var-scope -strict-whitespace -check-prefixes=GCN,GFX9,GFX89 %s

; GCN-LABEL: {{^}}lds_atomic_xchg_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw xchg ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xchg_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xchg ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xchg_ret_f64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_f64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr double, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xchg ptr addrspace(3) %gep, double 4.0 seq_cst
  store double %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xchg_ret_pointer_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_pointer_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr ptr, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xchg ptr addrspace(3) %gep, ptr null seq_cst
  store ptr %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_add_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw add ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add_ret_i64_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; SI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN: ds_add_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v[[[LOVDATA]]:[[HIVDATA]]] offset:32
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i64 4
  %result = atomicrmw add ptr addrspace(3) %gep, i64 9 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add1_ret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_add_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw add ptr addrspace(3) %ptr, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add1_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_add_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw add ptr addrspace(3) %gep, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw sub ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw sub ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub1_ret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_sub_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw sub ptr addrspace(3) %ptr, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub1_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw sub ptr addrspace(3) %gep, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_and_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_rtn_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw and ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_and_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw and ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_or_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_rtn_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw or ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_or_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw or ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xor_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_rtn_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw xor ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xor_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xor ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; FIXME: There is no atomic nand instr
; XGCN-LABEL: {{^}}lds_atomic_nand_ret_i64:uction, so we somehow need to expand this.
; define amdgpu_kernel void @lds_atomic_nand_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
;   %result = atomicrmw nand ptr addrspace(3) %ptr, i32 4 seq_cst
;   store i64 %result, ptr addrspace(1) %out, align 8
;   ret void
; }

; GCN-LABEL: {{^}}lds_atomic_min_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_rtn_i64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw min ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_min_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_rtn_i64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw min ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_max_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_rtn_i64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw max ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_max_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_rtn_i64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw max ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umin_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw umin ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umin_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw umin ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umax_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw umax ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umax_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw umax ptr addrspace(3) %gep, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xchg_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw xchg ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xchg_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xchg ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_add_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw add ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add_noret_i64_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; SI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN: ds_add_u64 {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]] offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i64 4
  %result = atomicrmw add ptr addrspace(3) %gep, i64 9 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add1_noret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_add_u64 {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw add ptr addrspace(3) %ptr, i64 1 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_add1_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_add_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw add ptr addrspace(3) %gep, i64 1 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw sub ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw sub ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub1_noret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_sub_u64 {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw sub ptr addrspace(3) %ptr, i64 1 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_sub1_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw sub ptr addrspace(3) %gep, i64 1 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_and_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw and ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_and_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw and ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_or_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw or ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_or_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw or ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xor_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_b64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw xor ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_xor_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_b64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw xor ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; FIXME: There is no atomic nand instr
; XGCN-LABEL: {{^}}lds_atomic_nand_noret_i64:uction, so we somehow need to expand this.
; define amdgpu_kernel void @lds_atomic_nand_noret_i64(ptr addrspace(3) %ptr) nounwind {
;   %result = atomicrmw nand ptr addrspace(3) %ptr, i32 4 seq_cst
;   ret void
; }

; GCN-LABEL: {{^}}lds_atomic_min_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_i64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw min ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_min_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_i64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw min ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_max_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_i64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw max ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_max_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_i64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw max ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umin_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw umin ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umin_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw umin ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umax_noret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_noret_i64(ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw umax ptr addrspace(3) %ptr, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_umax_noret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_noret_i64_offset(ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw umax ptr addrspace(3) %gep, i64 4 seq_cst
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_inc_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_inc_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw uinc_wrap ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc_ret_i64_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; SI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN: ds_inc_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v[[[LOVDATA]]:[[HIVDATA]]] offset:32
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_inc_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i64 4
  %result = atomicrmw uinc_wrap ptr addrspace(3) %gep, i64 9 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc1_ret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_inc_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_inc1_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw uinc_wrap ptr addrspace(3) %ptr, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_inc1_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_inc_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_inc1_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw uinc_wrap ptr addrspace(3) %gep, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_dec_ret_i64:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_dec_rtn_u64
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_dec_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw udec_wrap ptr addrspace(3) %ptr, i64 4 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_dec_ret_i64_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; SI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 9
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN: ds_dec_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v[[[LOVDATA]]:[[HIVDATA]]] offset:32
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_dec_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i64 4
  %result = atomicrmw udec_wrap ptr addrspace(3) %gep, i64 9 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_dec1_ret_i64:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 v[[LOVDATA:[0-9]+]], 1{{$}}
; GCN-DAG: v_mov_b32_e32 v[[HIVDATA:[0-9]+]], 0{{$}}
; GCN: ds_dec_rtn_u64 [[RESULT:v\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, v[[[LOVDATA]]:[[HIVDATA]]]
; GCN: buffer_store_dwordx2 [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_dec1_ret_i64(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %result = atomicrmw udec_wrap ptr addrspace(3) %ptr, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_dec1_ret_i64_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_dec_rtn_u64 {{.*}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_dec1_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %result = atomicrmw udec_wrap ptr addrspace(3) %gep, i64 1 seq_cst
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}
