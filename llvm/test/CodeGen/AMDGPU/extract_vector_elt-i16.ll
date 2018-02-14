; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=SICIVI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=SICIVI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn---amdgiz -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 %s

; GCN-LABEL: {{^}}extract_vector_elt_v2i16:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; GCN-DAG: v_mov_b32_e32 [[VELT0:v[0-9]+]], [[VEC]]
; GCN-DAG: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; GCN-DAG: buffer_store_short [[VELT0]]
; GCN-DAG: buffer_store_short [[VELT1]]
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
; GCN: s_lshl_b32 [[IDX_SCALED:s[0-9]+]], [[IDX]], 16
; GCN: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], [[IDX_SCALED]]
; GCN: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; GCN: buffer_store_short [[VELT1]]
; GCN: ScratchSize: 0
define amdgpu_kernel void @extract_vector_elt_v2i16_dynamic_sgpr(i16 addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, i32 %idx) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt = extractelement <2 x i16> %vec, i32 %idx
  store i16 %elt, i16 addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2i16_dynamic_vgpr:
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]]
; GCN-DAG: {{flat|buffer|global}}_load_dword [[IDX:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[IDX_SCALED:v[0-9]+]], 16, [[IDX]]

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
; GCN: buffer_load_ushort
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
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort
; SICIVI: buffer_store_short
; SICIVI: buffer_store_short

; GFX9-DAG: s_load_dword [[LOAD0:s[0-9]+]], s[0:1], 0x2c
; GFX9-DAG: s_load_dword [[LOAD1:s[0-9]+]], s[0:1], 0x30
; GFX9-DAG: v_mov_b32_e32 [[VLOAD0:v[0-9]+]], [[LOAD0]]
; GFX9-DAG: buffer_store_short [[VLOAD0]], off
; GFX9-DAG: v_mov_b32_e32 [[VLOAD1:v[0-9]+]], [[LOAD1]]
; GFX9-DAG: buffer_store_short [[VLOAD1]], off
define amdgpu_kernel void @extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 0
  %p1 = extractelement <4 x i16> %foo, i32 2
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 10
  store volatile i16 %p1, i16 addrspace(1)* %out, align 2
  store volatile i16 %p0, i16 addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3i16:
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort

; SICIVI: buffer_store_short
; SICIVI: buffer_store_short
; SICIVI: buffer_store_short

; SICIVI: buffer_load_ushort
; SICIVI: buffer_store_short

; GFX9: buffer_load_ushort
; GFX9: global_load_short_d16_hi
; GFX9: global_load_short_d16 v
; GFX9: buffer_store_dword
; GFX9: buffer_store_dword
; GFX9: buffer_load_ushort
; GFX9: buffer_store_short
define amdgpu_kernel void @dynamic_extract_vector_elt_v3i16(i16 addrspace(1)* %out, <3 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v4i16:
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort
; SICIVI: buffer_load_ushort

; SICIVI: buffer_store_short
; SICIVI: buffer_store_short
; SICIVI: buffer_store_short
; SICIVI: buffer_store_short

; SICIVI: buffer_load_ushort
; SICIVI: buffer_store_short

; GFX9: s_load_dword
; GFX9: buffer_store_dword
; GFX9: buffer_store_dword
; GFX9: buffer_load_ushort
; GFX9: buffer_store_short
define amdgpu_kernel void @dynamic_extract_vector_elt_v4i16(i16 addrspace(1)* %out, <4 x i16> %foo, i32 %idx) #0 {
  %p0 = extractelement <4 x i16> %foo, i32 %idx
  %out1 = getelementptr i16, i16 addrspace(1)* %out, i32 1
  store i16 %p0, i16 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
