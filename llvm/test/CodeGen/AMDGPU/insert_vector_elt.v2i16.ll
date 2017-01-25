; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=fiji -mattr=+flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=CIVI -check-prefix=VI %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=hawaii -mattr=+flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=CIVI -check-prefix=CI %s

; GCN-LABEL: {{^}}s_insertelement_v2i16_0:
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT1]], 0x3e7{{$}}
define void @s_insertelement_v2i16_0(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reg:
; GCN: s_load_dword [[ELT0:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI-DAG: s_and_b32 [[ELT0]], [[ELT0]], 0xffff{{$}}
; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]
define void @s_insertelement_v2i16_0_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi:
; GCN: s_load_dword [[ELT0:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]
define void @s_insertelement_v2i16_0_reghi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_1:
; GCN: s_load_dword [[VEC:s[0-9]+]]

; GCN-NOT: s_lshr
; GCN: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; GCN: s_or_b32 [[INS:s[0-9]+]], [[ELT0]], 0x3e70000
define void @s_insertelement_v2i16_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 1
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_1_reg:
; GCN: s_load_dword [[ELT1:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]

; GCN-NOT: shlr
define void @s_insertelement_v2i16_1_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 1
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2f16_0:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; CIVI: s_and_b32 [[ELT1:s[0-9]+]], [[VEC:s[0-9]+]], 0xffff0000
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT1]], 0x4500
define void @s_insertelement_v2f16_0(<2 x half> addrspace(1)* %out, <2 x half> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 0
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2f16_1:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN-NOT: s_lshr
; GCN: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; GCN: s_or_b32 [[INS:s[0-9]+]], [[ELT0]], 0x45000000
define void @s_insertelement_v2f16_1(<2 x half> addrspace(1)* %out, <2 x half> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 1
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_0:
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e7, [[ELT1]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2i16_0(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_0_reghi:
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: s_load_dword [[ELT0:s[0-9]+]]

; CIVI-DAG: s_lshr_b32 [[ELT0_SHIFT:s[0-9]+]], [[ELT0]], 16
; CIVI-DAG: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], [[ELT0_SHIFT]], [[ELT1]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2i16_0_reghi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, i32 %elt.arg) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_0_inlineimm:
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 53, [[ELT1]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2i16_0_inlineimm(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 53, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; FIXME: fold lshl_or c0, c1, v0 -> or (c0 << c1), v0

; GCN-LABEL: {{^}}v_insertelement_v2i16_1:
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e70000, [[VEC]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2i16_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 1
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_1_inlineimm:
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; GCN: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0xfff10000, [[ELT0]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2i16_1_inlineimm(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 -15, i32 1
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_0:
; GCN: flat_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x4500, [[ELT1]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2f16_0(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x half>, <2 x half> addrspace(1)* %in.gep
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 0
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_0_inlineimm:
; GCN: flat_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 53, [[ELT1]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2f16_0_inlineimm(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x half>, <2 x half> addrspace(1)* %in.gep
  %vecins = insertelement <2 x half> %vec, half 0xH0035, i32 0
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_1:
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x45000000, [[VEC]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2f16_1(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x half>, <2 x half> addrspace(1)* %in.gep
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 1
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_1_inlineimm:
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; GCN: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x230000, [[ELT0]]

; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define void @v_insertelement_v2f16_1_inlineimm(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x half>, <2 x half> addrspace(1)* %in.gep
  %vecins = insertelement <2 x half> %vec, half 0xH0023, i32 1
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out.gep
  ret void
}

; FIXME: Enable for others when argument load not split
; GCN-LABEL: {{^}}s_insertelement_v2i16_dynamic:
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7
; GCN: s_load_dword [[IDX:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VVEC:v[0-9]+]], [[VEC]]
; GCN-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 16
; GCN-DAG: s_lshl_b32 [[MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VVEC]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @s_insertelement_v2i16_dynamic(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 addrspace(2)* %idx.ptr) #0 {
  %idx = load volatile i32, i32 addrspace(2)* %idx.ptr
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 %idx
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_dynamic_sgpr:
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: s_load_dword [[IDX:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7
; GCN-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 16
; GCN-DAG: s_lshl_b32 [[MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VEC]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @v_insertelement_v2i16_dynamic_sgpr(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, i32 %idx) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 %idx
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_dynamic_vgpr:
; GCN: flat_load_dword [[IDX:v[0-9]+]]
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7

; VI-DAG: s_mov_b32 [[MASKK:s[0-9]+]], 0xffff{{$}}
; VI-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; VI: v_lshlrev_b32_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], [[MASKK]]

; CI: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; CI: v_lshl_b32_e32 [[MASK:v[0-9]+]], 0xffff, [[SCALED_IDX]]

; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VEC]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @v_insertelement_v2i16_dynamic_vgpr(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, i32 addrspace(1)* %idx.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %idx.gep = getelementptr inbounds i32, i32 addrspace(1)* %idx.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %idx = load i32, i32 addrspace(1)* %idx.gep
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 %idx
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_dynamic_vgpr:
; GCN: flat_load_dword [[IDX:v[0-9]+]]
; GCN: flat_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x1234

; VI-DAG: s_mov_b32 [[MASKK:s[0-9]+]], 0xffff{{$}}
; VI-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; VI: v_lshlrev_b32_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], [[MASKK]]

; CI: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; CI: v_lshl_b32_e32 [[MASK:v[0-9]+]], 0xffff, [[SCALED_IDX]]

; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VEC]]
; GCN: flat_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define void @v_insertelement_v2f16_dynamic_vgpr(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in, i32 addrspace(1)* %idx.ptr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %in, i64 %tid.ext
  %idx.gep = getelementptr inbounds i32, i32 addrspace(1)* %idx.ptr, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x half>, <2 x half> addrspace(1)* %out, i64 %tid.ext
  %idx = load i32, i32 addrspace(1)* %idx.gep
  %vec = load <2 x half>, <2 x half> addrspace(1)* %in.gep
  %vecins = insertelement <2 x half> %vec, half 0xH1234, i32 %idx
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
