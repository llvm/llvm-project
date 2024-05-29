; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}store_build_vector_multiple_uses_v4i32:
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword

; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4

; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
define amdgpu_kernel void @store_build_vector_multiple_uses_v4i32(ptr addrspace(1) noalias %out0,
                                                    ptr addrspace(1) noalias %out1,
                                                    ptr addrspace(1) noalias %out2,
                                                    ptr addrspace(1) %in) {
  %elt0 = load volatile i32, ptr addrspace(1) %in
  %elt1 = load volatile i32, ptr addrspace(1) %in
  %elt2 = load volatile i32, ptr addrspace(1) %in
  %elt3 = load volatile i32, ptr addrspace(1) %in

  %vec0 = insertelement <4 x i32> undef, i32 %elt0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %elt1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %elt2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %elt3, i32 3

  store <4 x i32> %vec3, ptr addrspace(1) %out0
  store <4 x i32> %vec3, ptr addrspace(1) %out1

  %extract0 = extractelement <4 x i32> %vec3, i32 0
  %extract1 = extractelement <4 x i32> %vec3, i32 1
  %extract2 = extractelement <4 x i32> %vec3, i32 2
  %extract3 = extractelement <4 x i32> %vec3, i32 3

  store volatile i32 %extract0, ptr addrspace(1) %out2
  store volatile i32 %extract1, ptr addrspace(1) %out2
  store volatile i32 %extract2, ptr addrspace(1) %out2
  store volatile i32 %extract3, ptr addrspace(1) %out2

  ret void
}

; GCN-LABEL: {{^}}store_build_vector_multiple_extract_uses_v4i32:
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword

; GCN: buffer_store_dwordx4

; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
; GCN: buffer_store_dword
define amdgpu_kernel void @store_build_vector_multiple_extract_uses_v4i32(ptr addrspace(1) noalias %out0,
                                                            ptr addrspace(1) noalias %out1,
                                                            ptr addrspace(1) noalias %out2,
                                                            ptr addrspace(1) %in) {
  %elt0 = load volatile i32, ptr addrspace(1) %in
  %elt1 = load volatile i32, ptr addrspace(1) %in
  %elt2 = load volatile i32, ptr addrspace(1) %in
  %elt3 = load volatile i32, ptr addrspace(1) %in

  %vec0 = insertelement <4 x i32> undef, i32 %elt0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %elt1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %elt2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %elt3, i32 3

  %extract0 = extractelement <4 x i32> %vec3, i32 0
  %extract1 = extractelement <4 x i32> %vec3, i32 1
  %extract2 = extractelement <4 x i32> %vec3, i32 2
  %extract3 = extractelement <4 x i32> %vec3, i32 3

  %op0 = add i32 %extract0, 3
  %op1 = sub i32 %extract1, 9
  %op2 = xor i32 %extract2, 1231412
  %op3 = and i32 %extract3, 258233412312

  store <4 x i32> %vec3, ptr addrspace(1) %out0

  store volatile i32 %op0, ptr addrspace(1) %out2
  store volatile i32 %op1, ptr addrspace(1) %out2
  store volatile i32 %op2, ptr addrspace(1) %out2
  store volatile i32 %op3, ptr addrspace(1) %out2

  ret void
}

; GCN-LABEL: {{^}}store_build_vector_multiple_uses_v4i32_bitcast_to_v2i64:
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword
; GCN: buffer_load_dword

; GCN: buffer_store_dwordx4

; GCN: buffer_store_dwordx2
; GCN: buffer_store_dwordx2
define amdgpu_kernel void @store_build_vector_multiple_uses_v4i32_bitcast_to_v2i64(ptr addrspace(1) noalias %out0,
                                                                     ptr addrspace(1) noalias %out1,
                                                                     ptr addrspace(1) noalias %out2,
                                                                     ptr addrspace(1) %in) {
  %elt0 = load volatile i32, ptr addrspace(1) %in
  %elt1 = load volatile i32, ptr addrspace(1) %in
  %elt2 = load volatile i32, ptr addrspace(1) %in
  %elt3 = load volatile i32, ptr addrspace(1) %in

  %vec0 = insertelement <4 x i32> undef, i32 %elt0, i32 0
  %vec1 = insertelement <4 x i32> %vec0, i32 %elt1, i32 1
  %vec2 = insertelement <4 x i32> %vec1, i32 %elt2, i32 2
  %vec3 = insertelement <4 x i32> %vec2, i32 %elt3, i32 3

  %bc.vec3 = bitcast <4 x i32> %vec3 to <2 x i64>
  store <2 x i64> %bc.vec3, ptr addrspace(1) %out0

  %extract0 = extractelement <2 x i64> %bc.vec3, i32 0
  %extract1 = extractelement <2 x i64> %bc.vec3, i32 1

  store volatile i64 %extract0, ptr addrspace(1) %out2
  store volatile i64 %extract1, ptr addrspace(1) %out2

  ret void
}
