; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}extract_vector_elt_v3f64_2:
; GCN: buffer_load_dwordx4
; GCN: buffer_load_dwordx2
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @extract_vector_elt_v3f64_2(double addrspace(1)* %out, <3 x double> addrspace(1)* %in) #0 {
  %ld = load volatile <3 x double>, <3 x double> addrspace(1)* %in
  %elt = extractelement <3 x double> %ld, i32 2
  store volatile double %elt, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v3f64:
; GCN-NOT: buffer_load
; SI-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; SI-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 2
; SI-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; SI: store_dwordx2 v[{{[0-9:]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; VI: s_cselect_b64 s{{\[}}[[T0LO:[0-9]+]]:[[T0HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}]
; VI: s_cmp_eq_u32 [[IDX:s[0-9]+]], 2
; VI: s_cselect_b64 s{{\[}}[[T1LO:[0-9]+]]:[[T1HI:[0-9]+]]{{\]}}, s[{{[0-9]+:[0-9]+}}], s{{\[}}[[T0LO]]:[[T0HI]]{{\]}}
; VI-DAG: v_mov_b32_e32 v[[V_LO:[0-9]+]], s[[T1LO]]
; VI-DAG: v_mov_b32_e32 v[[V_HI:[0-9]+]], s[[T1HI]]
; VI: store_dwordx2 v{{\[}}[[V_LO]]:[[V_HI]]{{\]}}
define amdgpu_kernel void @dyn_extract_vector_elt_v3f64(double addrspace(1)* %out, <3 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <3 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v4f64:
; GCN-NOT: buffer_load
; SI-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; SI-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 2
; SI-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; SI-DAG: s_cmp_eq_u32 [[IDX]], 3
; SI-DAG: s_cselect_b64 [[C3:[^,]+]], -1, 0
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; SI-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
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
define amdgpu_kernel void @dyn_extract_vector_elt_v4f64(double addrspace(1)* %out, <4 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <4 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
