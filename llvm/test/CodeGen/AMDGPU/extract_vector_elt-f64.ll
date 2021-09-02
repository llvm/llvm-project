; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

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
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 2
; GCN-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v3f64(double addrspace(1)* %out, <3 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <3 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dyn_extract_vector_elt_v4f64:
; GCN-NOT: buffer_load
; GCN-DAG: s_cmp_eq_u32 [[IDX:s[0-9]+]], 1
; GCN-DAG: s_cselect_b64 [[C1:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 2
; GCN-DAG: s_cselect_b64 [[C2:[^,]+]], -1, 0
; GCN-DAG: s_cmp_eq_u32 [[IDX]], 3
; GCN-DAG: s_cselect_b64 [[C3:[^,]+]], -1, 0
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C1]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C2]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; GCN-DAG: v_cndmask_b32_e{{32|64}} v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, [[C3]]
; GCN: store_dwordx2 v[{{[0-9:]+}}]
define amdgpu_kernel void @dyn_extract_vector_elt_v4f64(double addrspace(1)* %out, <4 x double> %foo, i32 %elt) #0 {
  %dynelt = extractelement <4 x double> %foo, i32 %elt
  store volatile double %dynelt, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
