; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}extract_vector_elt_v1i8:
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v1i8(i8 addrspace(1)* %out, <1 x i8> %foo) #0 {
  %p0 = extractelement <1 x i8> %foo, i32 0
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v2i8(i8 addrspace(1)* %out, <2 x i8> %foo) #0 {
  %p0 = extractelement <2 x i8> %foo, i32 0
  %p1 = extractelement <2 x i8> %foo, i32 1
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v3i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v3i8(i8 addrspace(1)* %out, <3 x i8> %foo) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 0
  %p1 = extractelement <3 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v4i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> %foo) #0 {
  %p0 = extractelement <4 x i8> %foo, i32 0
  %p1 = extractelement <4 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v8i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v8i8(i8 addrspace(1)* %out, <8 x i8> %foo) #0 {
  %p0 = extractelement <8 x i8> %foo, i32 0
  %p1 = extractelement <8 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v16i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v16i8(i8 addrspace(1)* %out, <16 x i8> %foo) #0 {
  %p0 = extractelement <16 x i8> %foo, i32 0
  %p1 = extractelement <16 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v32i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v32i8(i8 addrspace(1)* %out, <32 x i8> %foo) #0 {
  %p0 = extractelement <32 x i8> %foo, i32 0
  %p1 = extractelement <32 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v64i8:
; GCN: buffer_load_ubyte
; GCN: buffer_load_ubyte
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v64i8(i8 addrspace(1)* %out, <64 x i8> %foo) #0 {
  %p0 = extractelement <64 x i8> %foo, i32 0
  %p1 = extractelement <64 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p1, i8 addrspace(1)* %out
  store i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FIXME: SI generates much worse code from that's a pain to match

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v2i8:
; VI-DAG: buffer_load_ushort [[LOAD:v[0-9]+]],
; VI-DAG: s_load_dword [[IDX:s[0-9]+]], s[0:1], 0x30

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: v_lshrrev_b16_e32 [[EXTRACT:v[0-9]+]], [[SCALED_IDX]], [[LOAD]]
; VI: buffer_store_byte [[EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v2i8(i8 addrspace(1)* %out, <2 x i8> %foo, i32 %idx) #0 {
  %elt = extractelement <2 x i8> %foo, i32 %idx
  store i8 %elt, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3i8:
; VI-DAG: buffer_load_ubyte [[LOAD2:v[0-9]+]],
; VI-DAG: buffer_load_ushort [[LOAD01:v[0-9]+]],
; VI-DAG: s_load_dword [[IDX:s[0-9]+]], s[0:1], 0x30

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3

; VI: v_lshlrev_b32_e32 [[ELT2:v[0-9]+]], 16, [[LOAD2]]
; VI: v_or_b32_e32 [[VEC3:v[0-9]+]], [[LOAD01]], [[ELT2]]
; VI: v_lshrrev_b32_e32 [[EXTRACT:v[0-9]+]], [[SCALED_IDX]], [[VEC3]]
; VI: buffer_store_byte [[EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v3i8(i8 addrspace(1)* %out, <3 x i8> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v4i8:
; VI-DAG: s_load_dword [[VEC3:s[0-9]+]], s[0:1], 0x2c
; VI-DAG: s_load_dword [[IDX:s[0-9]+]], s[0:1], 0x30

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshr_b32 [[EXTRACT:s[0-9]+]], [[VEC3]], [[SCALED_IDX]]
; VI: v_mov_b32_e32 [[V_EXTRACT:v[0-9]+]], [[EXTRACT]]
; VI: buffer_store_byte [[V_EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> %foo, i32 %idx) #0 {
  %p0 = extractelement <4 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v8i8:
; VI-DAG: s_load_dwordx2 [[VEC3:s\[[0-9]+:[0-9]+\]]], s[0:1], 0x2c
; VI-DAG: s_load_dword [[IDX:s[0-9]+]], s[0:1], 0x34

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshr_b64 s{{\[}}[[EXTRACT_LO:[0-9]+]]:{{[0-9]+\]}}, [[VEC3]], [[SCALED_IDX]]
; VI: v_mov_b32_e32 [[V_EXTRACT:v[0-9]+]], s[[EXTRACT_LO]]
; VI: buffer_store_byte [[V_EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v8i8(i8 addrspace(1)* %out, <8 x i8> %foo, i32 %idx) #0 {
  %p0 = extractelement <8 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
