; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; How the replacement of i64 stores with v2i32 stores resulted in
; breaking other users of the bitcast if they already existed

; GCN-LABEL: {{^}}extract_vector_elt_select_error:
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dwordx2
define void @extract_vector_elt_select_error(i32 addrspace(1)* %out, i64 addrspace(1)* %in, i64 %val) nounwind {
  %vec = bitcast i64 %val to <2 x i32>
  %elt0 = extractelement <2 x i32> %vec, i32 0
  %elt1 = extractelement <2 x i32> %vec, i32 1

  store volatile i32 %elt0, i32 addrspace(1)* %out
  store volatile i32 %elt1, i32 addrspace(1)* %out
  store volatile i64 %val, i64 addrspace(1)* %in
  ret void
}
