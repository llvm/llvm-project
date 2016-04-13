; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure the add and load are reduced to 32-bits even with the
; bitcast to vector.
; GCN-LABEL: {{^}}bitcast_int_to_vector_extract_0:
; GCN-DAG: s_load_dword [[B:s[0-9]+]]
; GCN-DAG: buffer_load_dword [[A:v[0-9]+]]
; GCN: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, [[B]], [[A]]
; GCN: buffer_store_dword [[ADD]]
define void @bitcast_int_to_vector_extract_0(i32 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %b) {
   %a = load i64, i64 addrspace(1)* %in
   %add = add i64 %a, %b
   %val.bc = bitcast i64 %add to <2 x i32>
   %extract = extractelement <2 x i32> %val.bc, i32 0
   store i32 %extract, i32 addrspace(1)* %out
   ret void
}

; GCN-LABEL: {{^}}bitcast_fp_to_vector_extract_0:
; GCN: buffer_load_dwordx2
; GCN: v_add_f64
; GCN: buffer_store_dword v
define void @bitcast_fp_to_vector_extract_0(i32 addrspace(1)* %out, double addrspace(1)* %in, double %b) {
   %a = load double, double addrspace(1)* %in
   %add = fadd double %a, %b
   %val.bc = bitcast double %add to <2 x i32>
   %extract = extractelement <2 x i32> %val.bc, i32 0
   store i32 %extract, i32 addrspace(1)* %out
   ret void
}

; GCN-LABEL: {{^}}bitcast_int_to_fpvector_extract_0:
; GCN: buffer_load_dwordx2
; GCN: v_add_i32
; GCN: buffer_store_dword
define void @bitcast_int_to_fpvector_extract_0(float addrspace(1)* %out, i64 addrspace(1)* %in, i64 %b) {
   %a = load i64, i64 addrspace(1)* %in
   %add = add i64 %a, %b
   %val.bc = bitcast i64 %add to <2 x float>
   %extract = extractelement <2 x float> %val.bc, i32 0
   store float %extract, float addrspace(1)* %out
   ret void
}
