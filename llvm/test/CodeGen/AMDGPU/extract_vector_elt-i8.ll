; RUN: llc -mtriple=amdgcn-amd-amdhsa -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,SI %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -enable-var-scope -check-prefixes=GCN,VI %s

; GCN-LABEL: {{^}}extract_vector_elt_v1i8:
; GCN: s_load_dword [[LOAD:s[0-9]+]]
; GCN: v_mov_b32_e32 [[V_LOAD:v[0-9]+]], [[LOAD]]
; GCN: buffer_store_byte [[V_LOAD]]
define amdgpu_kernel void @extract_vector_elt_v1i8(i8 addrspace(1)* %out, <1 x i8> %foo) #0 {
  %p0 = extractelement <1 x i8> %foo, i32 0
  store i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i8:
; GCN: s_load_dword s
; GCN-NOT: {{flat|buffer|global}}
; SI: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
; VI: v_lshrrev_b16_e64 v{{[0-9]+}}, 8, s{{[0-9]+}}
; GCN-NOT: {{flat|buffer|global}}
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v2i8(i8 addrspace(1)* %out, <2 x i8> %foo) #0 {
  %p0 = extractelement <2 x i8> %foo, i32 0
  %p1 = extractelement <2 x i8> %foo, i32 1
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p1, i8 addrspace(1)* %out
  store volatile i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v3i8:
; GCN: s_load_dword s
; GCN-NOT: {{flat|buffer|global}}
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; GCN-NOT: {{flat|buffer|global}}
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v3i8(i8 addrspace(1)* %out, <3 x i8> %foo) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 0
  %p1 = extractelement <3 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p1, i8 addrspace(1)* %out
  store volatile i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v4i8:
; GCN: s_load_dword s
; GCN-NOT: {{flat|buffer|global}}
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; GCN-NOT: {{flat|buffer|global}}
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> %foo) #0 {
  %p0 = extractelement <4 x i8> %foo, i32 0
  %p1 = extractelement <4 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p1, i8 addrspace(1)* %out
  store volatile i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v8i8:
; GCN-NOT: {{s|flat|buffer|global}}_load
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN-NOT: {{s|flat|buffer|global}}_load
; GCN: s_lshr_b32 s{{[0-9]+}}, [[VAL]], 16
; GCN-NOT: {{s|flat|buffer|global}}_load
; GCN: buffer_store_byte
; GCN: buffer_store_byte
define amdgpu_kernel void @extract_vector_elt_v8i8(<8 x i8> %foo) #0 {
  %p0 = extractelement <8 x i8> %foo, i32 0
  %p1 = extractelement <8 x i8> %foo, i32 2
  store volatile i8 %p1, i8 addrspace(1)* null
  store volatile i8 %p0, i8 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v16i8:
; GCN: s_load_dword [[LOAD0:s[0-9]+]]
; GCN-NOT: {{flat|buffer|global}}
; GCN: s_lshr_b32 [[ELT2:s[0-9]+]], [[LOAD0]], 16
; GCN-DAG: v_mov_b32_e32 [[V_LOAD0:v[0-9]+]], [[LOAD0]]
; GCN-DAG: v_mov_b32_e32 [[V_ELT2:v[0-9]+]], [[ELT2]]
; GCN: buffer_store_byte [[V_ELT2]]
; GCN: buffer_store_byte [[V_LOAD0]]
define amdgpu_kernel void @extract_vector_elt_v16i8(i8 addrspace(1)* %out, <16 x i8> %foo) #0 {
  %p0 = extractelement <16 x i8> %foo, i32 0
  %p1 = extractelement <16 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p1, i8 addrspace(1)* %out
  store volatile i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v32i8:
; GCN-NOT: {{s|flat|buffer|global}}_load
; GCN: s_load_dword [[VAL:s[0-9]+]]
; GCN-NOT: {{s|flat|buffer|global}}_load
; GCN: s_lshr_b32 [[ELT2:s[0-9]+]], [[VAL]], 16
; GCN-DAG: v_mov_b32_e32 [[V_LOAD0:v[0-9]+]], s{{[0-9]+}}
; GCN-DAG: v_mov_b32_e32 [[V_ELT2:v[0-9]+]], [[ELT2]]
; GCN: buffer_store_byte [[V_ELT2]]
; GCN: buffer_store_byte [[V_LOAD0]]
define amdgpu_kernel void @extract_vector_elt_v32i8(<32 x i8> %foo) #0 {
  %p0 = extractelement <32 x i8> %foo, i32 0
  %p1 = extractelement <32 x i8> %foo, i32 2
  store volatile i8 %p1, i8 addrspace(1)* null
  store volatile i8 %p0, i8 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v64i8:
; GCN: s_load_dword [[LOAD0:s[0-9]+]]
; GCN-NOT: {{flat|buffer|global}}
; GCN: s_lshr_b32 [[ELT2:s[0-9]+]], [[LOAD0]], 16
; GCN-DAG: v_mov_b32_e32 [[V_LOAD0:v[0-9]+]], [[LOAD0]]
; GCN-DAG: v_mov_b32_e32 [[V_ELT2:v[0-9]+]], [[ELT2]]
; GCN: buffer_store_byte [[V_ELT2]]
; GCN: buffer_store_byte [[V_LOAD0]]
define amdgpu_kernel void @extract_vector_elt_v64i8(i8 addrspace(1)* %out, <64 x i8> %foo) #0 {
  %p0 = extractelement <64 x i8> %foo, i32 0
  %p1 = extractelement <64 x i8> %foo, i32 2
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p1, i8 addrspace(1)* %out
  store volatile i8 %p0, i8 addrspace(1)* %out1
  ret void
}

; FIXME: SI generates much worse code from that's a pain to match

; FIXME: 16-bit and 32-bit shift not combined after legalize to to
; isTypeDesirableForOp in SimplifyDemandedBits

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v2i8:
; VI: s_load_dword [[LOAD:s[0-9]+]], s[4:5], 0x28
; VI-NEXT: s_load_dword [[IDX:s[0-9]+]], s[4:5], 0x4c
; VI-NOT: {{flat|buffer|global}}
; VI-DAG: v_mov_b32_e32 [[V_LOAD:v[0-9]+]], [[LOAD]]
; VI-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: v_lshrrev_b16_e32 [[ELT:v[0-9]+]], [[SCALED_IDX]], [[V_LOAD]]
; VI: buffer_store_byte [[ELT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v2i8(i8 addrspace(1)* %out, [8 x i32], <2 x i8> %foo, [8 x i32], i32 %idx) #0 {
  %elt = extractelement <2 x i8> %foo, i32 %idx
  store volatile i8 %elt, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3i8:
; VI: s_load_dword [[LOAD:s[0-9]+]], s[4:5], 0x28
; VI-NEXT: s_load_dword [[IDX:s[0-9]+]], s[4:5], 0x4c
; VI-NOT: {{flat|buffer|global}}
; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshr_b32 [[ELT:s[0-9]+]], [[LOAD]], [[SCALED_IDX]]
; VI: v_mov_b32_e32 [[V_ELT:v[0-9]+]], [[ELT]]
; VI: buffer_store_byte [[V_ELT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v3i8(i8 addrspace(1)* %out, [8 x i32], <3 x i8> %foo, [8 x i32], i32 %idx) #0 {
  %p0 = extractelement <3 x i8> %foo, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v4i8:
; VI: s_load_dword [[IDX:s[0-9]+]], s[4:5], 0x30
; VI: s_load_dword [[VEC4:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshr_b32 [[EXTRACT:s[0-9]+]], [[VEC4]], [[SCALED_IDX]]

; VI: v_mov_b32_e32 [[V_EXTRACT:v[0-9]+]], [[EXTRACT]]
; VI: buffer_store_byte [[V_EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v4i8(i8 addrspace(1)* %out, <4 x i8> addrspace(4)* %vec.ptr, [8 x i32], i32 %idx) #0 {
  %vec = load <4 x i8>, <4 x i8> addrspace(4)* %vec.ptr
  %p0 = extractelement <4 x i8> %vec, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v8i8:
; VI: s_load_dword [[IDX:s[0-9]+]], s[4:5], 0x10
; VI: s_load_dwordx2 [[VEC8:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; VI: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 3
; VI: s_lshr_b64 s{{\[}}[[EXTRACT_LO:[0-9]+]]:{{[0-9]+\]}}, [[VEC8]], [[SCALED_IDX]]
; VI: v_mov_b32_e32 [[V_EXTRACT:v[0-9]+]], s[[EXTRACT_LO]]
; VI: buffer_store_byte [[V_EXTRACT]]
define amdgpu_kernel void @dynamic_extract_vector_elt_v8i8(i8 addrspace(1)* %out, <8 x i8> addrspace(4)* %vec.ptr, i32 %idx) #0 {
  %vec = load <8 x i8>, <8 x i8> addrspace(4)* %vec.ptr
  %p0 = extractelement <8 x i8> %vec, i32 %idx
  %out1 = getelementptr i8, i8 addrspace(1)* %out, i32 1
  store volatile i8 %p0, i8 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8i8_extract_0123:
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dword s
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 24
define amdgpu_kernel void @reduce_load_vector_v8i8_extract_0123() #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(4)* null
  %elt0 = extractelement <8 x i8> %load, i32 0
  %elt1 = extractelement <8 x i8> %load, i32 1
  %elt2 = extractelement <8 x i8> %load, i32 2
  %elt3 = extractelement <8 x i8> %load, i32 3
  store volatile i8 %elt0, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt1, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt2, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt3, i8 addrspace(1)* undef, align 1
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8i8_extract_0145:
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dwordx2
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
define amdgpu_kernel void @reduce_load_vector_v8i8_extract_0145() #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(4)* null
  %elt0 = extractelement <8 x i8> %load, i32 0
  %elt1 = extractelement <8 x i8> %load, i32 1
  %elt4 = extractelement <8 x i8> %load, i32 4
  %elt5 = extractelement <8 x i8> %load, i32 5
  store volatile i8 %elt0, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt1, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt4, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt5, i8 addrspace(1)* undef, align 1
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8i8_extract_45:
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_mov_b64 [[PTR:s\[[0-9]+:[0-9]+\]]], 4{{$}}
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0{{$}}
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
define amdgpu_kernel void @reduce_load_vector_v8i8_extract_45() #0 {
  %load = load <8 x i8>, <8 x i8> addrspace(4)* null
  %elt4 = extractelement <8 x i8> %load, i32 4
  %elt5 = extractelement <8 x i8> %load, i32 5
  store volatile i8 %elt4, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt5, i8 addrspace(1)* undef, align 1
  ret void
}

; FIXME: ought to be able to eliminate high half of load
; GCN-LABEL: {{^}}reduce_load_vector_v16i8_extract_0145:
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dwordx4
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 8
define amdgpu_kernel void @reduce_load_vector_v16i8_extract_0145() #0 {
  %load = load <16 x i8>, <16 x i8> addrspace(4)* null
  %elt0 = extractelement <16 x i8> %load, i32 0
  %elt1 = extractelement <16 x i8> %load, i32 1
  %elt4 = extractelement <16 x i8> %load, i32 4
  %elt5 = extractelement <16 x i8> %load, i32 5
  store volatile i8 %elt0, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt1, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt4, i8 addrspace(1)* undef, align 1
  store volatile i8 %elt5, i8 addrspace(1)* undef, align 1
  ret void
}

attributes #0 = { nounwind }
