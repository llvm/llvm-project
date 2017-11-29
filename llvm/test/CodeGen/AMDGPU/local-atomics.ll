; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,SICIVI,FUNC %s
; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,FUNC %s
; RUN: llc -march=r600 -mcpu=redwood -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=EG,FUNC %s

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32:
; EG: LDS_WRXCHG_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_load_dword [[SPTR:s[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_wrxchg_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_WRXCHG_RET *
; GCN: ds_wrxchg_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; XXX - Is it really necessary to load 4 into VGPR?
; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32:
; EG: LDS_ADD_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_load_dword [[SPTR:s[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_add_rtn_u32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_ADD_RET *
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_ret_i32_bad_si_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_ADD_RET *
; SI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_ret_i32:
; EG: LDS_ADD_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_ret_i32_offset:
; EG: LDS_ADD_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[ONE]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_ret_i32_bad_si_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_ADD_RET *
; SI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_add_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_ret_i32_bad_si_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_ret_i32:
; EG: LDS_SUB_RET *

; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_rtn_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_ret_i32_offset:
; EG: LDS_SUB_RET *

; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub1_ret_i32:
; EG: LDS_SUB_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_sub_rtn_u32  v{{[0-9]+}}, v{{[0-9]+}}, [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub1_ret_i32_offset:
; EG: LDS_SUB_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_sub_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, [[ONE]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32:
; EG: LDS_AND_RET *

; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_rtn_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_AND_RET *
; GCN: ds_and_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_OR_RET *
; GCN: ds_or_rtn_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_OR_RET *
; GCN: ds_or_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_XOR_RET *
; GCN: ds_xor_rtn_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_XOR_RET *
; GCN: ds_xor_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xor i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: There is no atomic nand instr
; XFUNC-LABEL: {{^}}lds_atomic_nand_ret_i32:uction, so we somehow need to expand this.
; define amdgpu_kernel void @lds_atomic_nand_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
;   %result = atomicrmw nand i32 addrspace(3)* %ptr, i32 4 seq_cst
;   store i32 %result, i32 addrspace(1)* %out, align 4
;   ret void
; }

; FUNC-LABEL: {{^}}lds_atomic_min_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MIN_INT_RET *
; GCN: ds_min_rtn_i32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MIN_INT_RET *
; GCN: ds_min_rtn_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MAX_INT_RET *
; GCN: ds_max_rtn_i32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MAX_INT_RET *
; GCN: ds_max_rtn_i32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MIN_UINT_RET *
; GCN: ds_min_rtn_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MIN_UINT_RET *
; GCN: ds_min_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MAX_UINT_RET *
; GCN: ds_max_rtn_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_ret_i32(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_ret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; EG: LDS_MAX_UINT_RET *
; GCN: ds_max_rtn_u32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_ret_i32_offset(i32 addrspace(1)* %out, i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  store i32 %result, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_load_dword [[SPTR:s[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_wrxchg_rtn_b32 [[RESULT:v[0-9]+]], [[VPTR]], [[DATA]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xchg i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xchg_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_wrxchg_rtn_b32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xchg_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xchg i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: s_load_dword [[SPTR:s[0-9]+]],
; GCN-DAG: v_mov_b32_e32 [[DATA:v[0-9]+]], 4
; GCN-DAG: v_mov_b32_e32 [[VPTR:v[0-9]+]], [[SPTR]]
; GCN: ds_add_u32 [[VPTR]], [[DATA]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add_noret_i32_bad_si_offset
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; SI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_noret_i32:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_add_u32 v{{[0-9]+}}, [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw add i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_noret_i32_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_add_u32 v{{[0-9]+}}, [[ONE]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_add1_noret_i32_bad_si_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; SI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}}
; CIVI: ds_add_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_add1_noret_i32_bad_si_offset(i32 addrspace(3)* %ptr, i32 %a, i32 %b) nounwind {
  %sub = sub i32 %a, %b
  %add = add i32 %sub, 4
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 %add
  %result = atomicrmw add i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_sub_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub1_noret_i32:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_sub_u32 v{{[0-9]+}}, [[ONE]]
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw sub i32 addrspace(3)* %ptr, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_sub1_noret_i32_offset:
; SICIVI-DAG: s_mov_b32 m0
; GFX9-NOT: m0

; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1{{$}}
; GCN: ds_sub_u32 v{{[0-9]+}}, [[ONE]] offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_sub1_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw sub i32 addrspace(3)* %gep, i32 1 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw and i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_and_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_and_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_and_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw and i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw or i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_or_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_or_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_or_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw or i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_b32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw xor i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_xor_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_xor_b32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_xor_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw xor i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FIXME: There is no atomic nand instr
; XFUNC-LABEL: {{^}}lds_atomic_nand_noret_i32:uction, so we somehow need to expand this.
; define amdgpu_kernel void @lds_atomic_nand_noret_i32(i32 addrspace(3)* %ptr) nounwind {
;   %result = atomicrmw nand i32 addrspace(3)* %ptr, i32 4 seq_cst
;   ret void
; }

; FUNC-LABEL: {{^}}lds_atomic_min_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_i32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw min i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_min_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_i32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_min_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw min i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_i32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw max i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_max_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_i32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_max_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw max i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umin i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umin_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_min_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umin_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umin i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_u32
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_noret_i32(i32 addrspace(3)* %ptr) nounwind {
  %result = atomicrmw umax i32 addrspace(3)* %ptr, i32 4 seq_cst
  ret void
}

; FUNC-LABEL: {{^}}lds_atomic_umax_noret_i32_offset:
; SICIVI: s_mov_b32 m0
; GFX9-NOT: m0

; GCN: ds_max_u32 v{{[0-9]+}}, v{{[0-9]+}} offset:16
; GCN: s_endpgm
define amdgpu_kernel void @lds_atomic_umax_noret_i32_offset(i32 addrspace(3)* %ptr) nounwind {
  %gep = getelementptr i32, i32 addrspace(3)* %ptr, i32 4
  %result = atomicrmw umax i32 addrspace(3)* %gep, i32 4 seq_cst
  ret void
}
