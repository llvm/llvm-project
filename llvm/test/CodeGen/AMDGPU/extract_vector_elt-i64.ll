; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,VI %s

; How the replacement of i64 stores with v2i32 stores resulted in
; breaking other users of the bitcast if they already existed

; GCN-LABEL: {{^}}extract_vector_elt_select_error:
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @extract_vector_elt_select_error(i32 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %val) #0 {
  %vec = bitcast i64 %val to <2 x i32>
  %elt0 = extractelement <2 x i32> %vec, i32 0
  %elt1 = extractelement <2 x i32> %vec, i32 1

  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i64 %val, i64 addrspace(1)* %in
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i64:
define amdgpu_kernel void @extract_vector_elt_v2i64(i64 addrspace(1)* %out, <2 x i64> %foo) #0 {
  %p0 = extractelement <2 x i64> %foo, i32 0
  %p1 = extractelement <2 x i64> %foo, i32 1
  %out1 = getelementptr i64, i64 addrspace(1)* %out, i32 1
  store volatile i64 %p1, i64 addrspace(1)* %out
  store volatile i64 %p0, i64 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v2i64:
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; SI-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI: store_dwordx2 v[{{[0-9:]+}}]
; VI: s_cselect_b64 s{{\[}}[[S_LO:[0-9]+]]:[[S_HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; VI-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[S_LO]]
; VI-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[S_HI]]
; VI: store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define amdgpu_kernel void @dyn_extract_vector_elt_v2i64(i64 addrspace(1)* %out, <2 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <2 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v2i64_2:
; GCN:     buffer_load_dwordx4
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v2i64_2(i64 addrspace(1)* %out, <2 x i64> addrspace(1)* %foo, i32 %elt, <2 x i64> %arst) #0 {
  %load = load volatile <2 x i64>, <2 x i64> addrspace(1)* %foo
  %or = or <2 x i64> %load, %arst
  %dynelt = extractelement <2 x i64> %or, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v3i64:
; SI-NOT: buffer_load
; SI-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; SI-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 2
; SI-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI: store_dwordx2 v[{{[0-9:]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; VI: s_cselect_b64 s{{\[}}[[T0LO:[0-9]+]]:[[T0HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 2
; VI: s_cselect_b64 s{{\[}}[[T1LO:[0-9]+]]:[[T1HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s{{\[}}[[T0LO]]:[[T0HI]]{{\]}}
; VI-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[T1LO]]
; VI-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[T1HI]]
; VI: store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define amdgpu_kernel void @dyn_extract_vector_elt_v3i64(i64 addrspace(1)* %out, <3 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <3 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v4i64:
; GCN-NOT: buffer_load
; SI-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; SI-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 2
; SI-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 3
; SI-DAG: s_cselect_b64 [[C3:[^,]+]], -1, 0
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; SI: store_dwordx2 v[{{[0-9:]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; VI: s_cselect_b64 s{{\[}}[[T0LO:[0-9]+]]:[[T0HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 2
; VI: s_cselect_b64 s{{\[}}[[T1LO:[0-9]+]]:[[T1HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s{{\[}}[[T0LO]]:[[T0HI]]{{\]}}
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 3
; VI: s_cselect_b64 s{{\[}}[[T2LO:[0-9]+]]:[[T2HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s{{\[}}[[T1LO]]:[[T1HI]]{{\]}}
; VI-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[T2LO]]
; VI-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[T2HI]]
; VI: store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define amdgpu_kernel void @dyn_extract_vector_elt_v4i64(i64 addrspace(1)* %out, <4 x i64> %foo, i32 %elt) #0 {
  %dynelt = extractelement <4 x i64> %foo, i32 %elt
  store volatile i64 %dynelt, i64 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
