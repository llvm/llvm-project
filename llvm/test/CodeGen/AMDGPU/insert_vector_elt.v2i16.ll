; RUN: llc -verify-machineinstrs -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -enable-amdgpu-aa=0 -mattr=+flat-for-global,-fp64-fp16-denormals < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9,GFX89 %s
; RUN: llc -verify-machineinstrs -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -enable-amdgpu-aa=0 -mattr=+flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,VI,GFX89 %s
; RUN: llc -verify-machineinstrs -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -enable-amdgpu-aa=0 -mattr=+flat-for-global < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CIVI,CI %s

; GCN-LABEL: {{^}}s_insertelement_v2i16_0:
; GCN: s_load_dword [[VEC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; CIVI: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT1]], 0x3e7{{$}}

; GFX9-NOT: lshr
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, 0x3e7, [[VEC]]
define amdgpu_kernel void @s_insertelement_v2i16_0(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reg:
; GCN-DAG: s_load_dword [[ELT_LOAD:s[0-9]+]], s[4:5],
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; CIVI-DAG: s_and_b32 [[ELT0:s[0-9]+]], [[ELT_LOAD]], 0xffff{{$}}
; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]

; GFX9-NOT: [[ELT0]]
; GFX9-NOT: [[VEC]]
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, [[ELT_LOAD]], [[VEC]]
define amdgpu_kernel void @s_insertelement_v2i16_0_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, [8 x i32], i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_multi_use_hi_reg:
; GCN-DAG: s_load_dword [[ELT_LOAD:s[0-9]+]], s[4:5],
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; CI-DAG: s_and_b32 [[ELT0_MASKED:s[0-9]+]], [[ELT_LOAD]], 0xffff{{$}}
; CI: s_lshr_b32 [[SHR:s[0-9]+]], [[VEC]], 16
; CI: s_lshl_b32 [[ELT1:s[0-9]+]], [[SHR]], 16
; CI-DAG: s_or_b32 s{{[0-9]+}}, [[ELT0_MASKED]], [[ELT1]]
; CI-DAG: ; use [[SHR]]


; FIXME: Should be able to void mask of upper bits
; VI-DAG: s_and_b32 [[ELT_MASKED:s[0-9]+]], [[ELT_LOAD]], 0xffff{{$}}
; VI-DAG: s_and_b32 [[VEC_HIMASK:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; VI-DAG: s_or_b32 [[OR:s[0-9]+]], [[ELT_MASKED]], [[VEC_HIMASK]]
; VI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VEC]], 16

; VI-DAG: ; use [[SHR]]


; GFX9: s_lshr_b32 [[ELT1:s[0-9]+]], [[VEC]], 16
; GFX9-DAG: s_pack_ll_b32_b16 s{{[0-9]+}}, [[ELT_LOAD]], [[ELT1]]
; GFX9-DAG: ; use [[ELT1]]
define amdgpu_kernel void @s_insertelement_v2i16_0_multi_use_hi_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, [8 x i32], i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt1 = extractelement <2 x i16> %vec, i32 1
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  %use1 = zext i16 %elt1 to i32
  call void asm sideeffect "; use $0", "s"(i32 %use1) #0
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi:
; GCN-DAG: s_load_dword [[ELT_ARG:s[0-9]+]], s[4:5],
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; CIVI: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; CIVI-DAG: s_and_b32 [[ELT1:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT_HI]], [[ELT1]]

; GFX9-NOT: [[ELT0]]
; GFX9-NOT: [[VEC]]
; GFX9: s_pack_hh_b32_b16 s{{[0-9]+}}, [[ELT_ARG]], [[VEC]]
define amdgpu_kernel void @s_insertelement_v2i16_0_reghi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, [8 x i32], i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi_multi_use_1:
; GCN: s_load_dword [[ELT_ARG:s[0-9]+]],
; GCN: s_load_dword [[VEC:s[0-9]+]],

; CIVI-DAG: s_lshr_b32 [[ELT1:s[0-9]+]], [[ELT_ARG]], 16
; CIVI-DAG: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff0000{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT1]], [[ELT0]]

; GFX9: s_lshr_b32 [[ELT1:s[0-9]+]], [[ELT_ARG]], 16
; GFX9: s_pack_lh_b32_b16 s{{[0-9]+}}, [[ELT1]], [[VEC]]
; GFX9: ; use [[ELT1]]
define amdgpu_kernel void @s_insertelement_v2i16_0_reghi_multi_use_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %elt.hi = lshr i32 %elt.arg, 16
  %elt = trunc i32 %elt.hi to i16
  %vecins = insertelement <2 x i16> %vec, i16 %elt, i32 0
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  %use1 = zext i16 %elt to i32
  call void asm sideeffect "; use $0", "s"(i32 %use1) #0
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_0_reghi_both_multi_use_1:
; GCN: s_load_dword [[ELT_ARG:s[0-9]+]],
; GCN: s_load_dword [[VEC:s[0-9]+]],

; CI-DAG: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; CI-DAG: s_lshr_b32 [[SHR:s[0-9]+]], [[VEC]], 16
; CI-DAG: s_lshl_b32 [[VEC_HI:s[0-9]+]], [[SHR]], 16
; CI: s_or_b32 s{{[0-9]+}}, [[ELT_HI]], [[VEC_HI]]


; VI-DAG: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; VI-DAG: s_lshr_b32 [[VEC_HI:s[0-9]+]], [[VEC]], 16
; VI: s_and_b32 [[MASK_HI:s[0-9]+]], [[VEC]], 0xffff0000
; VI: s_or_b32 s{{[0-9]+}}, [[ELT_HI]], [[MASK_HI]]

; GFX9-DAG: s_lshr_b32 [[ELT_HI:s[0-9]+]], [[ELT_ARG]], 16
; GFX9-DAG: s_lshr_b32 [[VEC_HI:s[0-9]+]], [[VEC]], 16
; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[ELT_HI]], [[VEC_HI]]
; GFX9: ; use [[ELT_HI]]
; GFX9: ; use [[VEC_HI]]
define amdgpu_kernel void @s_insertelement_v2i16_0_reghi_both_multi_use_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, i32 %elt.arg) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
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
define amdgpu_kernel void @s_insertelement_v2i16_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 1
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2i16_1_reg:
; GCN-DAG: s_load_dword [[ELT1_LOAD:s[0-9]+]], s[4:5],
; GCN-DAG: s_load_dword [[VEC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x0

; CIVI: s_lshl_b32 [[ELT1:s[0-9]+]], [[ELT1_LOAD]], 16
; CIVI: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; CIVI: s_or_b32 s{{[0-9]+}}, [[ELT0]], [[ELT1]]

; GCN-NOT: shlr
; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[VEC]], [[ELT1_LOAD]]
define amdgpu_kernel void @s_insertelement_v2i16_1_reg(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, [8 x i32], i16 %elt) #0 {
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
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
define amdgpu_kernel void @s_insertelement_v2f16_0(<2 x half> addrspace(1)* %out, <2 x half> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 0
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_insertelement_v2f16_1:
; GCN: s_load_dword [[VEC:s[0-9]+]]
; GCN-NOT: s_lshr

; CIVI: s_and_b32 [[ELT0:s[0-9]+]], [[VEC]], 0xffff{{$}}
; CIVI: s_or_b32 [[INS:s[0-9]+]], [[ELT0]], 0x45000000

; GFX9: s_pack_ll_b32_b16 s{{[0-9]+}}, [[VEC]], 0x4500
define amdgpu_kernel void @s_insertelement_v2f16_1(<2 x half> addrspace(1)* %out, <2 x half> addrspace(4)* %vec.ptr) #0 {
  %vec = load <2 x half>, <2 x half> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x half> %vec, half 5.000000e+00, i32 1
  store <2 x half> %vecins, <2 x half> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_0:
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]
; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e7, [[ELT1]]

; GFX9-DAG: s_movk_i32 [[ELT0:s[0-9]+]], 0x3e7{{$}}
; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 [[RES:v[0-9]+]], [[MASK]], [[ELT0]], [[VEC]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2i16_0(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
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
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: s_load_dword [[ELT0:s[0-9]+]]

; CIVI-DAG: s_lshr_b32 [[ELT0_SHIFT:s[0-9]+]], [[ELT0]], 16
; CIVI-DAG: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], [[ELT0_SHIFT]], [[ELT1]]

; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff0000{{$}}
; GFX9-DAG: v_lshrrev_b32_e64 [[ELT0_SHIFT:v[0-9]+]], 16, [[ELT0]]
; GFX9: v_and_or_b32 [[RES:v[0-9]+]], [[VEC]], [[MASK]], [[ELT0_SHIFT]]

; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2i16_0_reghi(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, i32 %elt.arg) #0 {
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
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 53, [[ELT1]]

; GFX9-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 [[RES:v[0-9]+]], [[MASK]], 53, [[VEC]]

; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2i16_0_inlineimm(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
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
; VI: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e70000
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x3e7
; GFX9-DAG: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[K]], 16, [[ELT0]]

; CI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, [[VEC]]
; CI: v_or_b32_e32 [[RES:v[0-9]+]], 0x3e70000, [[AND]]
; VI: v_or_b32_sdwa [[RES:v[0-9]+]], [[K]], [[VEC]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2i16_1(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
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
; VI: v_mov_b32_e32 [[K:v[0-9]+]], 0xfff10000
; GCN: {{flat|global}}_load_dword [[VEC:v[0-9]+]]
; CI:   v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; CI: v_or_b32_e32 [[RES:v[0-9]+]], 0xfff10000, [[ELT0]]
; VI: v_or_b32_sdwa [[RES:v[0-9]+]], [[K]], [[VEC]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], -15, 16, [[ELT0]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2i16_1_inlineimm(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
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
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 0x4500, [[ELT1]]

; GFX9-DAG: v_mov_b32_e32 [[ELT0:v[0-9]+]], 0x4500{{$}}
; GFX9-DAG: v_lshrrev_b32_e32 [[ELT1:v[0-9]+]], 16, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[ELT1]], 16, [[ELT0]]

; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2f16_0(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
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
; GCN: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; CIVI: v_and_b32_e32 [[ELT1:v[0-9]+]], 0xffff0000, [[VEC]]
; CIVI: v_or_b32_e32 [[RES:v[0-9]+]], 53, [[ELT1]]

; GFX9: v_lshrrev_b32_e32 [[ELT1:v[0-9]+]], 16, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[ELT1]], 16, 53
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2f16_0_inlineimm(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
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
; VI: v_mov_b32_e32 [[K:v[0-9]+]], 0x45000000
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; GFX9-DAG: s_movk_i32 [[K:s[0-9]+]], 0x4500
; GFX9-DAG: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], [[K]], 16, [[ELT0]]

; CI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, [[VEC]]
; CI: v_or_b32_e32 [[RES:v[0-9]+]], 0x45000000, [[AND]]

; VI: v_or_b32_sdwa [[RES:v[0-9]+]], [[K]], [[VEC]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2f16_1(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
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
; VI: v_mov_b32_e32 [[K:v[0-9]+]], 0x230000
; GCN: {{flat|global}}_load_dword [[VEC:v[0-9]+]]
; CI: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; GFX9: v_and_b32_e32 [[ELT0:v[0-9]+]], 0xffff, [[VEC]]
; CI: v_or_b32_e32 [[RES:v[0-9]+]], 0x230000, [[ELT0]]
; VI: v_or_b32_sdwa [[RES:v[0-9]+]], [[K]], [[VEC]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
; GFX9: v_lshl_or_b32 [[RES:v[0-9]+]], 35, 16, [[ELT0]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]]
define amdgpu_kernel void @v_insertelement_v2f16_1_inlineimm(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in) #0 {
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
; GCN-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 4
; GCN-DAG: s_lshl_b32 [[MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VVEC]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @s_insertelement_v2i16_dynamic(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(4)* %vec.ptr, i32 addrspace(4)* %idx.ptr) #0 {
  %idx = load volatile i32, i32 addrspace(4)* %idx.ptr
  %vec = load <2 x i16>, <2 x i16> addrspace(4)* %vec.ptr
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 %idx
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2i16_dynamic_sgpr:
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]
; GCN-DAG: s_load_dword [[IDX:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x3e7
; GCN-DAG: s_lshl_b32 [[SCALED_IDX:s[0-9]+]], [[IDX]], 4
; GCN-DAG: s_lshl_b32 [[MASK:s[0-9]+]], 0xffff, [[SCALED_IDX]]
; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VEC]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_insertelement_v2i16_dynamic_sgpr(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, i32 %idx) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %vecins = insertelement <2 x i16> %vec, i16 999, i32 %idx
  store <2 x i16> %vecins, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v2f16_dynamic_vgpr:
; GFX89-DAG: s_mov_b32 [[MASKK:s[0-9]+]], 0xffff{{$}}
; GCN-DAG: s_mov_b32 [[K:s[0-9]+]], 0x12341234

; GCN-DAG: {{flat|global}}_load_dword [[IDX:v[0-9]+]]
; GCN-DAG: {{flat|global}}_load_dword [[VEC:v[0-9]+]]

; GFX89-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 4, [[IDX]]
; GFX89-DAG: v_lshlrev_b32_e64 [[MASK:v[0-9]+]], [[SCALED_IDX]], [[MASKK]]

; CI-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 4, [[IDX]]
; CI-DAG: v_lshl_b32_e32 [[MASK:v[0-9]+]], 0xffff, [[SCALED_IDX]]

; GCN: v_bfi_b32 [[RESULT:v[0-9]+]], [[MASK]], [[K]], [[VEC]]
; GCN: {{flat|global}}_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RESULT]]
define amdgpu_kernel void @v_insertelement_v2f16_dynamic_vgpr(<2 x half> addrspace(1)* %out, <2 x half> addrspace(1)* %in, i32 addrspace(1)* %idx.ptr) #0 {
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

; GCN-LABEL: {{^}}v_insertelement_v4f16_0:
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]], s[4:5],
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GFX9-DAG: v_mov_b32_e32 [[BFI_MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 v[[INS_LO:[0-9]+]], [[BFI_MASK]], [[VAL]], v[[LO]]

; CIVI: s_and_b32 [[VAL_MASKED:s[0-9]+]], [[VAL]], 0xffff{{$}}
; CIVI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff0000, v[[LO]]
; CIVI: v_or_b32_e32 v[[INS_LO:[0-9]+]], [[VAL_MASKED]], [[AND]]

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[INS_LO]]:[[HI]]{{\]}}
define amdgpu_kernel void @v_insertelement_v4f16_0(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %in, [8 x i32], i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to half
  %vecins = insertelement <4 x half> %vec, half %val.cvt, i32 0
  store <4 x half> %vecins, <4 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4f16_1:
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GFX9: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, v[[LO]]
; GFX9: v_lshl_or_b32 v[[INS_HALF:[0-9]+]], [[VAL]], 16, [[AND]]

; VI: s_lshl_b32 [[VAL_HI:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[COPY_VAL:v[0-9]+]], [[VAL_HI]]
; VI: v_or_b32_sdwa v[[INS_HALF:[0-9]+]], [[COPY_VAL]], v[[LO]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

; CI: s_lshl_b32 [[VAL_HI:s[0-9]+]], [[VAL]], 16
; CI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, v[[LO]]
; CI: v_or_b32_e32 v[[INS_HALF:[0-9]+]], [[VAL_HI]], [[AND]]

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[INS_HALF]]:[[HI]]{{\]}}
define amdgpu_kernel void @v_insertelement_v4f16_1(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %in, i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to half
  %vecins = insertelement <4 x half> %vec, half %val.cvt, i32 1
  store <4 x half> %vecins, <4 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4f16_2:
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]], s[4:5],
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GFX9-DAG: v_mov_b32_e32 [[BFI_MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 v[[INS_HI:[0-9]+]], [[BFI_MASK]], [[VAL]], v[[HI]]

; CIVI: s_and_b32 [[VAL_MASKED:s[0-9]+]], [[VAL]], 0xffff{{$}}
; CIVI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff0000, v[[HI]]
; CIVI: v_or_b32_e32 v[[INS_HI:[0-9]+]], [[VAL_MASKED]], [[AND]]

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[INS_HI]]{{\]}}
define amdgpu_kernel void @v_insertelement_v4f16_2(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %in, [8 x i32], i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to half
  %vecins = insertelement <4 x half> %vec, half %val.cvt, i32 2
  store <4 x half> %vecins, <4 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4f16_3:
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GFX9: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, v[[HI]]
; GFX9: v_lshl_or_b32 v[[INS_HI:[0-9]+]], [[VAL]], 16, [[AND]]

; VI: s_lshl_b32 [[VAL_HI:s[0-9]+]], [[VAL]], 16
; VI-DAG: v_mov_b32_e32 [[COPY_VAL:v[0-9]+]], [[VAL_HI]]
; VI: v_or_b32_sdwa v[[INS_HI:[0-9]+]], [[COPY_VAL]], v[[HI]] dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0

; CI: s_lshl_b32 [[VAL_HI:s[0-9]+]], [[VAL]], 16
; CI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff, v[[HI]]
; CI: v_or_b32_e32 v[[INS_HI:[0-9]+]], [[VAL_HI]], [[AND]]

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[INS_HI]]{{\]}}
define amdgpu_kernel void @v_insertelement_v4f16_3(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %in, i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to half
  %vecins = insertelement <4 x half> %vec, half %val.cvt, i32 3
  store <4 x half> %vecins, <4 x half> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4i16_2:
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GFX9-DAG: v_mov_b32_e32 [[BFI_MASK:v[0-9]+]], 0xffff{{$}}
; GFX9: v_bfi_b32 v[[INS_HI:[0-9]+]], [[BFI_MASK]], [[VAL]], v[[HI]]

; CIVI: s_and_b32 [[VAL_MASKED:s[0-9]+]], [[VAL]], 0xffff{{$}}
; CIVI: v_and_b32_e32 [[AND:v[0-9]+]], 0xffff0000, v[[HI]]
; CIVI: v_or_b32_e32 v[[INS_HI:[0-9]+]], [[VAL_MASKED]], [[AND]]

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[}}[[LO]]:[[INS_HI]]{{\]}}
define amdgpu_kernel void @v_insertelement_v4i16_2(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in, i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to i16
  %vecins = insertelement <4 x i16> %vec, i16 %val.cvt, i32 2
  store <4 x i16> %vecins, <4 x i16> addrspace(1)* %out.gep
  ret void
}

; FIXME: Better code on CI?
; GCN-LABEL: {{^}}v_insertelement_v4i16_dynamic_vgpr:
; GCN-DAG: {{flat|global}}_load_dword [[IDX:v[0-9]+]],
; GCN-DAG: s_load_dword [[VAL:s[0-9]+]]
; GCN-DAG: {{flat|global}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}

; GCN-DAG: s_mov_b32 s[[MASK_LO:[0-9]+]], 0xffff
; GCN-DAG: s_mov_b32 s[[MASK_HI:[0-9]+]], 0
; CIVI-DAG: s_and_b32 [[MASKED_VAL:s[0-9]+]], [[VAL]], s[[MASK_LO]]
; VI-DAG: s_lshl_b32 [[SHIFTED_VAL:s[0-9]+]], [[MASKED_VAL]], 16
; CI-DAG: s_lshl_b32 [[SHIFTED_VAL:s[0-9]+]], [[VAL]], 16
; CIVI: s_or_b32 [[DUP_VAL:s[0-9]+]], [[MASKED_VAL]], [[SHIFTED_VAL]]
; GCN-DAG: v_lshlrev_b32_e32 [[SCALED_IDX:v[0-9]+]], 4, [[IDX]]
; GFX9-DAG: s_pack_ll_b32_b16 [[DUP_VAL:s[0-9]+]], [[VAL]], [[VAL]]
; GFX89: v_lshlrev_b64 v[{{[0-9:]+}}], [[SCALED_IDX]], s{{\[}}[[MASK_LO]]:[[MASK_HI]]{{\]}}
; CI: v_lshl_b64 v[{{[0-9:]+}}], s[{{[0-9:]+}}], [[SCALED_IDX]]
; GCN: v_bfi_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[DUP_VAL]], v{{[0-9]+}}
; GCN: v_bfi_b32 v{{[0-9]+}}, v{{[0-9]+}}, [[DUP_VAL]], v{{[0-9]+}}

; GCN: {{flat|global}}_store_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}
define amdgpu_kernel void @v_insertelement_v4i16_dynamic_vgpr(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in, i32 %val) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %out, i64 %tid.ext
  %idx.val = load volatile i32, i32 addrspace(1)* undef
  %vec = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to i16
  %vecins = insertelement <4 x i16> %vec, i16 %val.cvt, i32 %idx.val
  store <4 x i16> %vecins, <4 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_insertelement_v4f16_dynamic_sgpr:
define amdgpu_kernel void @v_insertelement_v4f16_dynamic_sgpr(<4 x half> addrspace(1)* %out, <4 x half> addrspace(1)* %in, i32 %val, i32 %idxval) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x half>, <4 x half> addrspace(1)* %out, i64 %tid.ext
  %vec = load <4 x half>, <4 x half> addrspace(1)* %in.gep
  %val.trunc = trunc i32 %val to i16
  %val.cvt = bitcast i16 %val.trunc to half
  %vecins = insertelement <4 x half> %vec, half %val.cvt, i32 %idxval
  store <4 x half> %vecins, <4 x half> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
