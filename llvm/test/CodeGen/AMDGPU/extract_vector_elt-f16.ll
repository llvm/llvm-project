; RUN: llc -march=amdgcn -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mtriple=amdgcn-- -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}extract_vector_elt_v2f16:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; GCN-DAG: v_mov_b32_e32 [[VELT0:v[0-9]+]], [[VEC]]
; GCN-DAG: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; GCN-DAG: buffer_store_short [[VELT0]]
; GCN-DAG: buffer_store_short [[VELT1]]
define amdgpu_kernel void @extract_vector_elt_v2f16(half addrspace(1)* %out, <2 x half> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(4)* %vec.ptr
  %p0 = extractelement <2 x half> %vec, i32 0
  %p1 = extractelement <2 x half> %vec, i32 1
  %out1 = getelementptr half, half addrspace(1)* %out, i32 10
  store half %p1, half addrspace(1)* %out, align 2
  store half %p0, half addrspace(1)* %out1, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2f16_dynamic_sgpr:
; GCN: s_load_dword [[IDX:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN: s_lshl_b32 [[IDX_SCALED:s[0-9]+]], [[IDX]], 4
; GCN: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], [[IDX_SCALED]]
; GCN: v_mov_b32_e32 [[VELT1:v[0-9]+]], [[ELT1]]
; GCN: buffer_store_short [[VELT1]]
; GCN: ScratchSize: 0
define amdgpu_kernel void @extract_vector_elt_v2f16_dynamic_sgpr(half addrspace(1)* %out, <2 x half> addrspace(4)* %vec.ptr, i32 %idx) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(4)* %vec.ptr
  %elt = extractelement <2 x half> %vec, i32 %idx
  store half %elt, half addrspace(1)* %out, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v2f16_dynamic_vgpr:
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]]
; GCN-DAG: {{flat|buffer}}_load_dword [[IDX:v[0-9]+]]
; GCN: v_lshlrev_b32_e32 [[IDX_SCALED:v[0-9]+]], 4, [[IDX]]

; SI: v_lshr_b32_e32 [[ELT:v[0-9]+]], [[VEC]], [[IDX_SCALED]]
; VI: v_lshrrev_b32_e64 [[ELT:v[0-9]+]], [[IDX_SCALED]], [[VEC]]


; SI: buffer_store_short [[ELT]]
; VI: flat_store_short v{{\[[0-9]+:[0-9]+\]}}, [[ELT]]
; GCN: ScratchSize: 0{{$}}
define amdgpu_kernel void @extract_vector_elt_v2f16_dynamic_vgpr(half addrspace(1)* %out, <2 x half> addrspace(4)* %vec.ptr, i32 addrspace(1)* %idx.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %idx.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x half>, <2 x half> addrspace(4)* %vec.ptr
  %idx = load i32, i32 addrspace(1)* %gep
  %elt = extractelement <2 x half> %vec, i32 %idx
  store half %elt, half addrspace(1)* %out.gep, align 2
  ret void
}

; GCN-LABEL: {{^}}extract_vector_elt_v3f16:
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2

; GCN: buffer_store_short
; GCN: buffer_store_short
define amdgpu_kernel void @extract_vector_elt_v3f16(half addrspace(1)* %out, <3 x half> %foo) #0 {
  %p0 = extractelement <3 x half> %foo, i32 0
  %p1 = extractelement <3 x half> %foo, i32 2
  %out1 = getelementptr half, half addrspace(1)* %out, i32 1
  store half %p1, half addrspace(1)* %out, align 2
  store half %p0, half addrspace(1)* %out1, align 2
  ret void
}

; FIXME: Why sometimes vector shift?
; GCN-LABEL: {{^}}dynamic_extract_vector_elt_v3f16:
; SI: s_load_dword s
; SI: s_load_dwordx2 s
; SI: s_load_dwordx2 s

; GFX89: s_load_dwordx2 s
; GFX89: s_load_dwordx2 s
; GFX89: s_load_dword s


; GCN-DAG: s_lshl_b32 s{{[0-9]+}}, s{{[0-9]+}}, 4
; GCN: s_lshr_b64 s{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}

; GCN: {{buffer|global}}_store_short
define amdgpu_kernel void @dynamic_extract_vector_elt_v3f16(half addrspace(1)* %out, <3 x half> %foo, i32 %idx) #0 {
  %p0 = extractelement <3 x half> %foo, i32 %idx
  %out1 = getelementptr half, half addrspace(1)* %out, i32 1
  store half %p0, half addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_extractelement_v4f16_2:
; SI: buffer_load_dword [[LOAD:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: buffer_store_short [[LOAD]]

; VI: flat_load_dword v
; VI: flat_store_short

; GFX9: global_load_dword [[LOAD:v[0-9]+]], v{{\[[0-9]+:[0-9]+\]}}, off offset:4
; GFX9: global_store_short_d16_hi v{{\[[0-9]+:[0-9]+\]}}, [[LOAD]]
define amdgpu_kernel void @v_extractelement_v4f16_2(half addrspace(1)* %out, <4 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %vec.extract = extractelement <4 x half> %vec, i32 2
  store half %vec.extract, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4f16_dynamic_vgpr:
; GCN-DAG: {{flat|global|buffer}}_load_dword [[IDX:v[0-9]+]],
; GCN-DAG: {{flat|global|buffer}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 4, [[IDX]]

; GFX89: v_lshrrev_b64 v{{\[}}[[SHIFT_LO:[0-9]+]]:[[SHIFT_HI:[0-9]+]]{{\]}}, [[SCALED_IDX]], v{{\[}}[[LO]]:[[HI]]{{\]}}
; GFX89: {{flat|global}}_store_short v{{\[[0-9]+:[0-9]+\]}}, v[[SHIFT_LO]]

; SI: v_lshr_b64 v{{\[}}[[SHIFT_LO:[0-9]+]]:[[SHIFT_HI:[0-9]+]]{{\]}}, v{{\[}}[[LO]]:[[HI]]{{\]}}, [[SCALED_IDX]]
; SI: buffer_store_short v[[SHIFT_LO]]
define amdgpu_kernel void @v_insertelement_v4f16_dynamic_vgpr(half addrspace(1)* %out, <4 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds half, half addrspace(1)* %out, i64 %tid.ext
  %idx.val = load volatile i32, i32 addrspace(1)* undef
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %vec.extract = extractelement <4 x half> %vec, i32 %idx.val
  store half %vec.extract, half addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8f16_extract_01:
; GCN: s_load_dwordx2 [[PTR:s\[[0-9]+:[0-9]+\]]],
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], 0x0
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
define amdgpu_kernel void @reduce_load_vector_v8f16_extract_01(<16 x half> addrspace(4)* %ptr) #0 {
  %load = load <16 x half>, <16 x half> addrspace(4)* %ptr
  %elt0 = extractelement <16 x half> %load, i32 0
  %elt1 = extractelement <16 x half> %load, i32 1
  store volatile half %elt0, half addrspace(1)* undef, align 2
  store volatile half %elt1, half addrspace(1)* undef, align 2
  ret void
}

; GCN-LABEL: {{^}}reduce_load_vector_v8f16_extract_23:
; GCN: s_load_dwordx2 [[PTR:s\[[0-9]+:[0-9]+\]]],
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_load_dword s{{[0-9]+}}, [[PTR]], {{0x1|0x4}}
; GCN-NOT: {{s|buffer|flat|global}}_load_
; GCN: s_lshr_b32 s{{[0-9]+}}, s{{[0-9]+}}, 16
define amdgpu_kernel void @reduce_load_vector_v8f16_extract_23(<16 x half> addrspace(4)* %ptr) #0 {
  %load = load <16 x half>, <16 x half> addrspace(4)* %ptr
  %elt2 = extractelement <16 x half> %load, i32 2
  %elt3 = extractelement <16 x half> %load, i32 3
  store volatile half %elt2, half addrspace(1)* undef, align 2
  store volatile half %elt3, half addrspace(1)* undef, align 2
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
