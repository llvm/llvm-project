; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=gfx901 -mattr=+flat-for-global,-fp64-fp16-denormals < %s | FileCheck -check-prefix=GCN -check-prefix=GFX9 -check-prefix=GFX89 %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=fiji -mattr=+flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=CIVI -check-prefix=VI -check-prefix=GFX89 %s
; RUN: llc -verify-machineinstrs -march=amdgcn -mcpu=hawaii -mattr=+flat-for-global < %s | FileCheck -check-prefix=GCN -check-prefix=CIVI -check-prefix=CI %s

; GCN-LABEL: {{^}}s_insertelement_v2i16_0:
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT1]], 0x3e7{{$}}

; GFX9-NOT: lshr
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, 0x3e7, [[VEC]]
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

; GFX9-NOT: [[ELT0]]
; GFX9-NOT: [[VEC]]
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, [[ELT0]], [[VEC]]
define void @s_insertelement_v2i16_0_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_multi_use_hi_reg:
; GCN: s_load_dword [[ELT0:s[0-9]+]]
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI-DAG: s_and_b32 [[ELT0]], [[ELT0]], 0xffff{{$}}
; CIVI: s_lshr_b32 [[SHR:s[0-9]+]], [[VEC]], 16
; CIVI: s_lshl_b32 [[ELT1:s[0-9]+]], [[SHR]], 16
; CIVI-DAG: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]
; CIVI-DAG: ; use [[SHR]]

; GFX9: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; GFX9-DAG: s_pack_ll_b32_b16 s{{[0-9]+}}, [[ELT0]], [[ELT1]]
; GFX9-DAG: ; use [[ELT1]]
define void @s_insertelement_v2i16_0_multi_use_hi_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt1 = extractelement <2 x i16> %vec, i32 1
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  %use1 = zext i16 %elt1 to i32
  call void asm sideeffect "; use $0", "s"(i32 %use1) #0
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi:
; GCN: s_load_dword [[ELT_ARG:s[0-9]+]], s[0:1]
; GCN: s_load_dword [[VEC:s[0-9]+]]

; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT_ARG]], [[ELT1]]

; GFX9-NOT: [[ELT0]]
; GFX9-NOT: [[VEC]]
; GFX9: s_pack_hh_b32_b16 s{{[0-9]+}}, [[ELT_ARG]], [[VEC]]
define void @s_insertelement_v2i16_0_reghi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi_multi_use_1:
; GCN: s_load_dword [[ELT_ARG:s[0-9]+]], s[0:1]
; GCN: s_load_dword [[VEC:s[0-9]+]],

; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]

; GFX9: s_lshr_b32 [[ELT1:s[0-9]+]], [[ELT_ARG]], 16
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, [[ELT1]], [[VEC]]
; GFX9: ; use [[ELT1]]
define void @s_insertelement_v2i16_0_reghi_multi_use_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  %use1 = zext i16 %elt to i32
  call void asm sideeffect "; use $0", "s"(i32 %use1) #0
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi_both_multi_use_1:
; GCN: s_load_dword [[ELT_ARG:s[0-9]+]], s[0:1]
; GCN: s_load_dword [[VEC:s[0-9]+]],

; CIVI-DAG: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; CIVI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VEC]], 16
; CIVI-DAG: s_lshl_b32 [[VEC_HI:s[0-9]+]], [[SHR]], 16
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT_HI]], [[VEC_HI]]

; GFX9-DAG: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; GFX9-DAG: s_lshr_b32 [[VEC_HI:s[0-9]+]], [[VEC]], 16
; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[ELT_HI]], [[VEC_HI]]
; GFX9: ; use [[ELT_HI]]
; GFX9: ; use [[VEC_HI]]
define void @s_insertelement_v2i16_0_reghi_both_multi_use_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(2)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(2)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vec.hi = extractelement <2 x i16> %vec, i32 1
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  %use1 = zext i16 %elt to i32
  %vec.hi.use1 = zext i16 %vec.hi to i32

  call void asm sideeffect "; use $0", "s"(i32 %use1) #0
  call void asm sideeffect "; use $0", "s"(i32 %vec.hi.use1) #0
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_1:
; GCN: s_load_dword [[VEC:s[0-9]+]]

; GCN-NOT: s_lshr

; CIVI: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; CIVI: s_or_b32 [[INS:s[0-9]+]], [[ELT0]], 0x3e70000

; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[VEC]], 0x3e7
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
; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[VEC]], [[ELT1]]
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

; GFX9: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, 0x4500, [[ELT1]]
define void @s_insertelement_v2f16_0(<2 x half> addrspace(1)* %out, <2 x half> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 0
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2f16_1:
; GFX9: s_load_dword [[VEC:s[0-9]+]]
; GCN-NOT: s_lshr

; CIVI: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; CIVI: s_or_b32 [[INS:s[0-9]+]], [[ELT0]], 0x45000000

; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[VEC]], 0x4500
define void @s_insertelement_v2f16_1(<2 x half> addrspace(1)* %out, <2 x half> addrspace(2)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(2)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 1
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_0:
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e7, [[ELT1]]

; GFX9-DAG: s_movk_i32 [[ELT0:s[0-9]+]], 0x3e7{{$}}
; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 [[RES:v[0-9]+]], [[MASK]], [[ELT0]], [[VEC]]
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

; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff{{$}}
; GFX9-DAG: v_lshrrev_b32_e64 [[ELT0_SHIFT:v[0-9]+]], 16, [[ELT0]]
; GFX9: v_and_or_b32 [[RES:v[0-9]+]], [[VEC]], [[MASK]], [[ELT0_SHIFT]]

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

; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 [[RES:v[0-9]+]], [[MASK]], 53, [[VEC]]

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
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e70000, [[VEC]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x3e7
; GFX9-DAG: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[K]], 16, [[ELT0]]

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
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], -15, 16, [[ELT0]]
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
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x4500, [[ELT1]]

; GFX9-DAG: v_mov_b32_e32 [[ELT0:v[0-9]+]], 0x4500{{$}}
; GFX9-DAG: v_lshrrev_b32_e32 [[ELT1:v[0-9]+]], 16, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[ELT1]], 16, [[ELT0]]

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

; GFX9: v_lshrrev_b32_e32 [[ELT1:v[0-9]+]], 16, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[ELT1]], 16, 53
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
; GCN-DAG: flat_load_dword [[VEC:v[0-9]+]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x45000000, [[VEC]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x4500
; GFX9-DAG: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[K]], 16, [[ELT0]]

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
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], 35, 16, [[ELT0]]
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

; GFX89-DAG: s_mov_b32 [[MASKK:s[0-9]+]], 0xffff{{$}}
; GFX89-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; GFX89-DAG: v_lshlrev_b32_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], [[MASKK]]

; CI-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; CI-DAG: v_lshl_b32_e32 [[MASK:v[0-9]+]], 0xffff, [[SCALED_IDX]]

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

; GFX89-DAG: s_mov_b32 [[MASKK:s[0-9]+]], 0xffff{{$}}
; GFX89-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; GFX89-DAG: v_lshlrev_b32_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], [[MASKK]]

; CI-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 16, [[IDX]]
; CI-DAG: v_lshl_b32_e32 [[MASK:v[0-9]+]], 0xffff, [[SCALED_IDX]]

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
