; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefixes=SI,SICI,SICIVI,PREGFX11,GCN %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefixes=SICI,CIVI,SICIVI,PREGFX11,GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefixes=CIVI,SICIVI,GFX8PLUS,PREGFX11,GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9PLUS,GFX8PLUS,PREGFX11,GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1100 -amdgpu-enable-vopd=0 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX11,GFX9PLUS,GFX8PLUS,GCN %s

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_offset:
; GFX9PLUS-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x13
; SICI-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x1c
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x70
; GCN-DAG: v_mov_b32_e32 [[VCMP:v[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; PREGFX11: ds_cmpst_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[VCMP]], [[VSWAP]] offset:16
; GFX11: ds_cmpstore_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[VSWAP]], [[VCMP]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i32_offset(ptr addrspace(1) %out, [8 x i32], ptr addrspace(3) %ptr, [8 x i32], i32 %swap) nounwind {
  %gep = getelementptr i32, ptr addrspace(3) %ptr, i32 4
  %pair = cmpxchg ptr addrspace(3) %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i64_offset:
; GFX9PLUS-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SICI-DAG: s_load_dwordx2 s[[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GFX8PLUS-DAG: s_load_{{dwordx2|b64}} s[[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0x34
; GCN-DAG: v_mov_b32_e32 v[[LOVCMP:[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 v[[HIVCMP:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; GCN-DAG: v_mov_b32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; PREGFX11: ds_cmpst_rtn_b64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v[[[LOVCMP]]:[[HIVCMP]]], v[[[LOSWAPV]]:[[HISWAPV]]] offset:32
; GFX11: ds_cmpstore_rtn_b64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VPTR]], v[[[LOSWAPV]]:[[HISWAPV]]], v[[[LOVCMP]]:[[HIVCMP]]] offset:32
; GCN: [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i64_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %pair = cmpxchg ptr addrspace(3) %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  store i64 %result, ptr addrspace(1) %out, align 8
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_ret_i32_bad_si_offset
; GFX9PLUS-NOT: m0
; SI: ds_cmpst_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_cmpst_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GFX9PLUS: ds_{{cmpst|cmpstore}}_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_ret_i32_bad_si_offset(ptr addrspace(1) %out, ptr addrspace(3) %ptr, i32 %swap, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, ptr addrspace(3) %ptr, i32 %add
  %pair = cmpxchg ptr addrspace(3) %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  store i32 %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_noret_i32_offset:
; GFX9PLUS-NOT: m0
; SICIVI-DAG: s_mov_b32 m0


; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SICI-DAG: s_load_dword [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x12
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[SWAP:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x48
; GCN-DAG: v_mov_b32_e32 [[VCMP:v[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 [[VSWAP:v[0-9]+]], [[SWAP]]
; PREGFX11: ds_cmpst_b32 [[VPTR]], [[VCMP]], [[VSWAP]] offset:16
; GFX11: ds_cmpstore_b32 [[VPTR]], [[VSWAP]], [[VCMP]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_noret_i32_offset(ptr addrspace(3) %ptr, [8 x i32], i32 %swap) nounwind {
  %gep = getelementptr i32, ptr addrspace(3) %ptr, i32 4
  %pair = cmpxchg ptr addrspace(3) %gep, i32 7, i32 %swap seq_cst monotonic
  %result = extractvalue { i32, i1 } %pair, 0
  ret void
}

; GCN-LABEL: {{^}}lds_atomic_cmpxchg_noret_i64_offset:
; GFX9PLUS-NOT: m0
; SICIVI-DAG: s_mov_b32 m0

; SICI-DAG: s_load_dword [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; SICI-DAG: s_load_dwordx2 s[[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; GFX8PLUS-DAG: s_load_{{dword|b32}} [[PTR:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GFX8PLUS-DAG: s_load_{{dwordx2|b64}} s[[[LOSWAP:[0-9]+]]:[[HISWAP:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, 0x2c
; GCN-DAG: v_mov_b32_e32 v[[LOVCMP:[0-9]+]], 7
; GCN-DAG: v_mov_b32_e32 v[[HIVCMP:[0-9]+]], 0
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[PTR]]
; GCN-DAG: v_mov_b32_e32 v[[LOSWAPV:[0-9]+]], s[[LOSWAP]]
; GCN-DAG: v_mov_b32_e32 v[[HISWAPV:[0-9]+]], s[[HISWAP]]
; PREGFX11: ds_cmpst_b64 [[VPTR]], v[[[LOVCMP]]:[[HIVCMP]]], v[[[LOSWAPV]]:[[HISWAPV]]] offset:32
; GFX11: ds_cmpstore_b64 [[VPTR]], v[[[LOSWAPV]]:[[HISWAPV]]], v[[[LOVCMP]]:[[HIVCMP]]] offset:32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_cmpxchg_noret_i64_offset(ptr addrspace(3) %ptr, i64 %swap) nounwind {
  %gep = getelementptr i64, ptr addrspace(3) %ptr, i32 4
  %pair = cmpxchg ptr addrspace(3) %gep, i64 7, i64 %swap seq_cst monotonic
  %result = extractvalue { i64, i1 } %pair, 0
  ret void
}
