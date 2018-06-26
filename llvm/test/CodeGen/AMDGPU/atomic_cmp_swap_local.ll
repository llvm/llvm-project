; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SI,SICI,SICIVI,GCN %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SICI,CIVI,SICIVI,GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI,CIVI,SICIVI,GFX89,GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX9,GFX89,GCN %s

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_offset:
; GFX9-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x13
; SICI-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x1c
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; GFX89-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x70
; GCN-DAG: v_mov_b32_e32 [[VCMP:v[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; GCN: ds_cmpst_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[VCMP]], [[VSWAP]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i32_offset(i32 addrspace(1)* %out, [8 x i32], i32 addrspace(3)* %ptr, [8 x i32], i32 %swap) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i64_offset:
; GFX9-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SICI-DAG: s_load_dwordx2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xd
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GFX89-DAG: s_load_dwordx2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x34
; GCN-DAG: v_mov_b32_e32 v[[LOVCMP:[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 v[[HIVCMP:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; GCN-DAG: v_mov_b32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; GCN: ds_cmpst_rtn_b64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v{{\[}}[[LOVCMP]]:[[HIVCMP]]{{\]}}, v{{\[}}[[LOSWAPV]]:[[HISWAPV]]{{\]}} offset:32
; GCN: [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i64_offset(i64 addrspace(1)* %out, i64 addrspace(3)* %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64, i64 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i64 addrspace(3)* %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  store i64 %result, i64 addrspace(1)* %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_bad_si_offset
; GFX9-NOT: m0
; SI: ds_cmpst_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_cmpst_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GFX9: ds_cmpst_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %swap, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 %add
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_noret_i32_offset:
; GFX9-NOT: m0
; SICIVI-DAG: s_mov_b32 m0


; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SICI-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x12
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GFX89-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x48
; GCN-DAG: v_mov_b32_e32 [[VCMP:v[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; GCN: ds_cmpst_b32 [[VPTR]], [[VCMP]], [[VSWAP]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_noret_i32_offset(i32 addrspace(3)* %ptr, [8 x i32], i32 %swap) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i32 addrspace(3)* %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_noret_i64_offset:
; GFX9-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SICI-DAG: s_load_dwordx2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0xb
; GFX89-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GFX89-DAG: s_load_dwordx2 s{{\[}}[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-DAG: v_mov_b32_e32 v[[LOVCMP:[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 v[[HIVCMP:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; GCN-DAG: v_mov_b32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; GCN: ds_cmpst_b64 [[VPTR]], v{{\[}}[[LOVCMP]]:[[HIVCMP]]{{\]}}, v{{\[}}[[LOSWAPV]]:[[HISWAPV]]{{\]}} offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_noret_i64_offset(i64 addrspace(3)* %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64, i64 addrspace(3)* %ptr, i32 4
  %pair = cmpxchg i64 addrspace(3)* %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  ret void
}
