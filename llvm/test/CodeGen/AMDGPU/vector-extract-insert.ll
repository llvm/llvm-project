; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Test that when extracting the same unknown vector index from an
; insertelement the dynamic indexing is folded away.

declare i32 @llvm.amdgcn.workitem.id.x() #0

; No dynamic indexing required
; GCN-LABEL: {{^}}extract_insert_same_dynelt_v4i32:
; GCN: s_load_dword [[VAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd{{$}}
; GCN-NOT buffer_load_dword
; GCN-NOT: [[VAL]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOT: [[VVAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @extract_insert_same_dynelt_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %val, i32 %idx) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep.in = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in, i64 %id.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %id.ext
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %gep.in
  %insert = insertelement <4 x i32> %vec, i32 %val, i32 %idx
  %extract = extractelement <4 x i32> %insert, i32 %idx
  store i32 %extract, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}extract_insert_different_dynelt_v4i32:
; GCN: buffer_load_dwordx4
; GCN: v_movreld_b32
; GCN: v_movrels_b32
; GCN: buffer_store_dword v
define void @extract_insert_different_dynelt_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %val, i32 %idx0, i32 %idx1) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep.in = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in, i64 %id.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %id.ext
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %gep.in
  %insert = insertelement <4 x i32> %vec, i32 %val, i32 %idx0
  %extract = extractelement <4 x i32> %insert, i32 %idx1
  store i32 %extract, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}extract_insert_same_elt2_v4i32:
; GCN: s_load_dword [[VAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd{{$}}
; GCN-NOT buffer_load_dword
; GCN-NOT: [[VAL]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOT: [[VVAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @extract_insert_same_elt2_v4i32(i32 addrspace(1)* %out, <4 x i32> addrspace(1)* %in, i32 %val, i32 %idx) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep.in = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %in, i64 %id.ext
  %gep.out = getelementptr inbounds i32, i32 addrspace(1)* %out, i64 %id.ext
  %vec = load <4 x i32>, <4 x i32> addrspace(1)* %gep.in
  %insert = insertelement <4 x i32> %vec, i32 %val, i32 %idx
  %extract = extractelement <4 x i32> %insert, i32 %idx
  store i32 %extract, i32 addrspace(1)* %gep.out
  ret void
}

; GCN-LABEL: {{^}}extract_insert_same_dynelt_v4f32:
; GCN: s_load_dword [[VAL:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd{{$}}
; GCN-NOT buffer_load_dword
; GCN-NOT: [[VAL]]
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOT: [[VVAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @extract_insert_same_dynelt_v4f32(float addrspace(1)* %out, <4 x float> addrspace(1)* %in, float %val, i32 %idx) #1 {
  %id = call i32 @llvm.amdgcn.workitem.id.x()
  %id.ext = sext i32 %id to i64
  %gep.in = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %in, i64 %id.ext
  %gep.out = getelementptr inbounds float, float addrspace(1)* %out, i64 %id.ext
  %vec = load volatile <4 x float>, <4 x float> addrspace(1)* %gep.in
  %insert = insertelement <4 x float> %vec, float %val, i32 %idx
  %extract = extractelement <4 x float> %insert, i32 %idx
  store float %extract, float addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }