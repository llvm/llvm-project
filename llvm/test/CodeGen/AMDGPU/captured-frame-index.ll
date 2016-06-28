; RUN: llc -march=amdgcn -mattr=-promote-alloca -amdgpu-sroa=0 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}stored_fi_to_lds:
; GCN: s_load_dword [[LDSPTR:s[0-9]+]]
; GCN: v_mov_b32_e32 [[ZERO1:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, [[ZERO1]]
; GCN: v_mov_b32_e32 [[ZERO0:v[0-9]+]], 0{{$}}
; GCN: v_mov_b32_e32 [[VLDSPTR:v[0-9]+]], [[LDSPTR]]
; GCN: ds_write_b32  [[VLDSPTR]], [[ZERO0]]
define void @stored_fi_to_lds(float* addrspace(3)* %ptr) #0 {
  %tmp = alloca float
  store float 4.0, float *%tmp
  store float* %tmp, float* addrspace(3)* %ptr
  ret void
}

; Offset is applied
; GCN-LABEL: {{^}}stored_fi_to_lds_2_small_objects:
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}
; GCN-DAG: buffer_store_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:4{{$}}

; GCN-DAG: s_load_dword [[LDSPTR:s[0-9]+]]

; GCN-DAG: v_mov_b32_e32 [[VLDSPTR:v[0-9]+]], [[LDSPTR]]
; GCN: ds_write_b32  [[VLDSPTR]], [[ZERO]]

; GCN-DAG: v_mov_b32_e32 [[FI1:v[0-9]+]], 4{{$}}
; GCN: ds_write_b32  [[VLDSPTR]], [[FI1]]
define void @stored_fi_to_lds_2_small_objects(float* addrspace(3)* %ptr) #0 {
  %tmp0 = alloca float
  %tmp1 = alloca float
  store float 4.0, float* %tmp0
  store float 4.0, float* %tmp1
  store volatile float* %tmp0, float* addrspace(3)* %ptr
  store volatile float* %tmp1, float* addrspace(3)* %ptr
  ret void
}

; Same frame index is used multiple times in the store
; GCN-LABEL: {{^}}stored_fi_to_self:
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x4d2{{$}}
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[K]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}
; GCN: buffer_store_dword [[ZERO]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}
define void @stored_fi_to_self() #0 {
  %tmp = alloca i32*

  ; Avoid optimizing everything out
  store volatile i32* inttoptr (i32 1234 to i32*), i32** %tmp
  %bitcast = bitcast i32** %tmp to i32*
  store volatile i32* %bitcast, i32** %tmp
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_self_offset:
; GCN-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN-DAG: v_mov_b32_e32 [[K0:v[0-9]+]], 32{{$}}
; GCN: buffer_store_dword [[K0]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}

; GCN-DAG: v_mov_b32_e32 [[K1:v[0-9]+]], 0x4d2{{$}}
; GCN: buffer_store_dword [[K1]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:2048{{$}}

; GCN: v_mov_b32_e32 [[OFFSETK:v[0-9]+]], 0x800{{$}}
; GCN: buffer_store_dword [[OFFSETK]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:2048{{$}}
define void @stored_fi_to_self_offset() #0 {
  %tmp0 = alloca [512 x i32]
  %tmp1 = alloca i32*

  ; Avoid optimizing everything out
  %tmp0.cast = bitcast [512 x i32]* %tmp0 to i32*
  store volatile i32 32, i32* %tmp0.cast

  store volatile i32* inttoptr (i32 1234 to i32*), i32** %tmp1

  %bitcast = bitcast i32** %tmp1 to i32*
  store volatile i32* %bitcast, i32** %tmp1
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_fi:
; GCN: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:4{{$}}
; GCN: buffer_store_dword v{{[0-9]+}}, [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:8{{$}}

; GCN: v_mov_b32_e32 [[FI1:v[0-9]+]], 4{{$}}
; GCN: buffer_store_dword [[FI1]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:8{{$}}

; GCN: v_mov_b32_e32 [[FI2:v[0-9]+]], 8{{$}}
; GCN: buffer_store_dword [[FI2]], [[ZERO]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen offset:4{{$}}
define void @stored_fi_to_fi() #0 {
  %tmp0 = alloca i32*
  %tmp1 = alloca i32*
  %tmp2 = alloca i32*
  store volatile i32* inttoptr (i32 1234 to i32*), i32** %tmp0
  store volatile i32* inttoptr (i32 5678 to i32*), i32** %tmp1
  store volatile i32* inttoptr (i32 9999 to i32*), i32** %tmp2

  %bitcast1 = bitcast i32** %tmp1 to i32*
  %bitcast2 = bitcast i32** %tmp2 to i32* ;  at offset 8

  store volatile i32* %bitcast1, i32** %tmp2 ; store offset 4 at offset 8
  store volatile i32* %bitcast2, i32** %tmp1 ; store offset 8 at offset 4
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_global:
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
; GCN: v_mov_b32_e32 [[FI:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[FI]]
define void @stored_fi_to_global(float* addrspace(1)* %ptr) #0 {
  %tmp = alloca float
  store float 0.0, float *%tmp
  store float* %tmp, float* addrspace(1)* %ptr
  ret void
}

; Offset is applied
; GCN-LABEL: {{^}}stored_fi_to_global_2_small_objects:
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen
; GCN: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen

; GCN: v_mov_b32_e32 [[FI1:v[0-9]+]], 4{{$}}
; GCN: buffer_store_dword [[FI1]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}

; GCN-DAG: v_mov_b32_e32 [[FI2:v[0-9]+]], 8{{$}}
; GCN: buffer_store_dword [[FI2]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
define void @stored_fi_to_global_2_small_objects(float* addrspace(1)* %ptr) #0 {
  %tmp0 = alloca float
  %tmp1 = alloca float
  %tmp2 = alloca float
  store volatile float 0.0, float *%tmp0
  store volatile float 0.0, float *%tmp1
  store volatile float 0.0, float *%tmp2
  store volatile float* %tmp1, float* addrspace(1)* %ptr
  store volatile float* %tmp2, float* addrspace(1)* %ptr
  ret void
}

; GCN-LABEL: {{^}}stored_fi_to_global_huge_frame_offset:
; GCN: s_add_i32 [[BASE_1_OFF_0:s[0-9]+]], 0, 0x3ffc
; GCN: v_mov_b32_e32 [[BASE_0:v[0-9]+]], 0{{$}}
; GCN: buffer_store_dword [[BASE_0]], v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen

; GCN: v_mov_b32_e32 [[V_BASE_1_OFF_0:v[0-9]+]], [[BASE_1_OFF_0]]
; GCN: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7{{$}}
; GCN: s_add_i32 [[BASE_1_OFF_1:s[0-9]+]], 0, 56
; GCN: buffer_store_dword [[K]], [[V_BASE_1_OFF_0]], s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offen{{$}}

; GCN: v_mov_b32_e32 [[V_BASE_1_OFF_1:v[0-9]+]], [[BASE_1_OFF_1]]
; GCN: buffer_store_dword [[V_BASE_1_OFF_1]], off, s{{\[[0-9]+:[0-9]+\]}}, 0{{$}}
define void @stored_fi_to_global_huge_frame_offset(i32* addrspace(1)* %ptr) #0 {
  %tmp0 = alloca [4096 x i32]
  %tmp1 = alloca [4096 x i32]
  %gep0.tmp0 = getelementptr [4096 x i32], [4096 x i32]* %tmp0, i32 0, i32 0
  store volatile i32 0, i32* %gep0.tmp0
  %gep1.tmp0 = getelementptr [4096 x i32], [4096 x i32]* %tmp0, i32 0, i32 4095
  store volatile i32 999, i32* %gep1.tmp0
  %gep0.tmp1 = getelementptr [4096 x i32], [4096 x i32]* %tmp0, i32 0, i32 14
  store i32* %gep0.tmp1, i32* addrspace(1)* %ptr
  ret void
}

attributes #0 = { nounwind }
