; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,SI,SIVI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,GFX89,SIVI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s

; GCN-LABEL: {{^}}extract_vector_elt_v2i16:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; SIVI: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; SIVI-DAG: v_mov_b32_e32 [[VELT0:v[0-9]+]], [[VEC]]
; SIVI-DAG: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; SIVI-DAG: buffer_store_short [[VELT0]]
; SIVI-DAG: buffer_store_short [[VELT1]]
; GFX9: v_mov_b32_e32 [[VVEC:v[0-9]+]], [[VEC]]
; GFX9: global_store_short_d16_hi v{{[0-9]+}}, [[VVEC]],
; GFX9: buffer_store_short [[VVEC]],
define amdgpu_kernel void @extract_vector_elt_v2i16(i16 addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %p0 = extractelement <2 x i16> %vec, i32 0
  %p1 = extractelement <2 x i16> %vec, i32 1
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i16_dynamic_sgpr:
; GCN: s_load_dword [[IDX:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN: s_lshl_b32 [[IDX_SCALED:s[0-9]+]], [[IDX]], 4
; GCN: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], [[IDX_SCALED]]
; GCN: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; GCN: buffer_store_short [[VELT1]]
; GCN: ScratchSize: 0
define amdgpu_kernel void @extract_vector_elt_v2i16_dynamic_sgpr(i16 addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, [8 x i32], i32 %idx) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt = extractelement <2 x i16> %vec, i32 %idx
  store i16 %elt, i16 addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i16_dynamic_vgpr:
; GCN-DAG: {{flat|buffer|global}}_load_dword [[IDX:v[0-9]+]]
; GCN-DAG: v_lshlrev_b32_e32 [[IDX_SCALED:v[0-9]+]], 4, [[IDX]]
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]]

; SI: v_lshr_b32_e32 [[ELT:v[0-9]+]], [[VEC]], [[IDX_SCALED]]
; VI: v_lshrrev_b32_e64 [[ELT:v[0-9]+]], [[IDX_SCALED]], [[VEC]]

; SI: buffer_store_short [[ELT]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[ELT]]
; GCN: ScratchSize: 0{{$}}
define amdgpu_kernel void @extract_vector_elt_v2i16_dynamic_vgpr(i16 addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, i32 addrspace(1)* %idx.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %idx.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 %tid.ext
  %idx = load volatile i32, i32 addrspace(1)* %gep
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt = extractelement <2 x i16> %vec, i32 %idx
  store i16 %elt, i16 addrspace(1)* %out.gep, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v3i16:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2

; GCN-NOT: {{buffer|flat|global}}_load

; GCN: buffer_store_short
; GCN: buffer_store_short
define amdgpu_kernel void @extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 0
  %p1 = extractelement <3 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p1, i16 addrspace(1)* %out, align 2
  store i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v4i16:
; SI: s_load_dwordx2
; SI: buffer_store_short
; SI: buffer_store_short

; GFX89-DAG: s_load_dwordx2 s{{\[}}[[LOAD0:[0-9]+]]:[[LOAD1:[0-9]+]]{{\]}}, s[0:1], 0x2c
; GFX89-DAG: v_mov_b32_e32 [[VLOAD0:v[0-9]+]], s[[LOAD0]]
; GFX89-DAG: buffer_store_short [[VLOAD0]], off
; GFX89-DAG: v_mov_b32_e32 [[VLOAD1:v[0-9]+]], s[[LOAD1]]
; GFX89-DAG: buffer_store_short [[VLOAD1]], off
define amdgpu_kernel void @extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 0
  %p1 = extractelement <4 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store volatile i16 %p1, i16 addrspace(1)* %out, align 2
  store volatile i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3i16:
; SI: s_load_dword s
; SI: s_load_dwordx2 s
; SI: s_load_dwordx2 s

; GFX89-DAG: s_load_dwordx2 s{{\[}}[[LOAD0:[0-9]+]]:[[LOAD1:[0-9]+]]{{\]}}, s[0:1], 0x24
; GFX89-DAG: s_load_dwordx2 s{{\[}}[[LOAD0:[0-9]+]]:[[LOAD1:[0-9]+]]{{\]}}, s[0:1], 0x4c
; GFX89-DAG: s_load_dword s{{[0-9]+}}, s[0:1], 0x54

; GCN-NOT: {{buffer|flat|global}}

; SICI: buffer_store_short
; SICI: buffer_store_short
; SICI: buffer_store_short

; GFX9-NOT: s_pack_ll_b32_b16
; GFX9-NOT: s_pack_lh_b32_b16

; GCN-DAG: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 4
; GCN: s_lshr_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s
; GCN: {{buffer|global}}_store_short
define amdgpu_kernel void @dynamic_extract_vector_elt_v3i16(i16 addrspace(1)* %out, [8 x i32], <3 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4i16_dynamic_sgpr:
define amdgpu_kernel void @v_insertelement_v4i16_dynamic_sgpr(i16 addrspace(1)* %out, <4 x i16> addrspace(1)* %in, i32 %idx) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds i16, i16 addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep
  %vec.extract = extractelement <4 x i16> %vec, i32 %idx
  store i16 %vec.extract, i16 addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8i16_extract_01:
; GCN: s_load_dwordx2 [[PTR:s\[[0-9]+:[0-9]+\]]],
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
define amdgpu_kernel void @reduce_load_vector_v8i16_extract_01(<16 x i16> addrspace(4)* %ptr) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(4)* %ptr
  %elt0 = extractelement <16 x i16> %load, i32 0
  %elt1 = extractelement <16 x i16> %load, i32 1
  store volatile i16 %elt0, i16 addrspace(1)* undef, align 2
  store volatile i16 %elt1, i16 addrspace(1)* undef, align 2
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8i16_extract_23:
; GCN: s_load_dwordx2 [[PTR:s\[[0-9]+:[0-9]+\]]],
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], {{0x1|0x4}}
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
define amdgpu_kernel void @reduce_load_vector_v8i16_extract_23(<16 x i16> addrspace(4)* %ptr) #0 {
  %load = load <16 x i16>, <16 x i16> addrspace(4)* %ptr
  %elt2 = extractelement <16 x i16> %load, i32 2
  %elt3 = extractelement <16 x i16> %load, i32 3
  store volatile i16 %elt2, i16 addrspace(1)* undef, align 2
  store volatile i16 %elt3, i16 addrspace(1)* undef, align 2
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
