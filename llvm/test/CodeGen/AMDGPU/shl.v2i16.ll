; RUN: llc -march=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,VI,CIVI %s
; RUN: llc -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,CI,CIVI %s

; GCN-LABEL: {{^}}s_shl_v2i16:
; GFX9: s_load_dword [[LHS:s[0-9]+]]
; GFX9: s_load_dword [[RHS:s[0-9]+]]
; GFX9: v_mov_b32_e32 [[VLHS:v[0-9]+]], [[LHS]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], [[RHS]], [[VLHS]]

; VI: s_load_dword s
; VI: s_load_dword s
; VI: s_lshr_b32
; VI: s_lshr_b32
; VI: s_and_b32
; VI: s_and_b32
; VI: s_and_b32
; VI: s_or_b32


; CI: s_load_dword s
; CI: s_load_dword s
; CI: s_lshr_b32
; CI: s_and_b32
; CI: s_lshr_b32
; CI: s_lshl_b32
; CI: s_lshl_b32
; CI: s_lshl_b32
; CI: s_and_b32
; CI: s_or_b32
; CI: _store_dword
define amdgpu_kernel void @s_shl_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> %lhs, <2 x i16> %rhs) #0 {
  %result = shl <2 x i16> %lhs, %rhs
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_shl_v2i16:
; GCN: {{buffer|flat|global}}_load_dword [[LHS:v[0-9]+]]
; GCN: {{buffer|flat|global}}_load_dword [[RHS:v[0-9]+]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], [[RHS]], [[LHS]]

; VI: v_lshlrev_b16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_lshlrev_b16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; CI: s_mov_b32 [[MASK:s[0-9]+]], 0xffff{{$}}
; CI: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, [[LHS]]
; CI: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; CI: v_lshlrev_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_lshl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, [[MASK]], v{{[0-9]+}}
; CI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_shl_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %b_ptr = getelementptr <2 x i16>, <2 x i16> addrspace(1)* %in.gep, i32 1
  %a = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %b = load <2 x i16>, <2 x i16> addrspace(1)* %b_ptr
  %result = shl <2 x i16> %a, %b
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}shl_v_s_v2i16:
; GFX9: s_load_dword [[RHS:s[0-9]+]]
; GFX9: {{buffer|flat|global}}_load_dword [[LHS:v[0-9]+]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], [[RHS]], [[LHS]]
define amdgpu_kernel void @shl_v_s_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, <2 x i16> %sgpr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vgpr = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %result = shl <2 x i16> %vgpr, %sgpr
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}shl_s_v_v2i16:
; GFX9: s_load_dword [[LHS:s[0-9]+]]
; GFX9: {{buffer|flat|global}}_load_dword [[RHS:v[0-9]+]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], [[RHS]], [[LHS]]
define amdgpu_kernel void @shl_s_v_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in, <2 x i16> %sgpr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vgpr = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %result = shl <2 x i16> %sgpr, %vgpr
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}shl_imm_v_v2i16:
; GCN: {{buffer|flat|global}}_load_dword [[RHS:v[0-9]+]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], [[RHS]], 8
define amdgpu_kernel void @shl_imm_v_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vgpr = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %result = shl <2 x i16> <i16 8, i16 8>, %vgpr
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}shl_v_imm_v2i16:
; GCN: {{buffer|flat|global}}_load_dword [[LHS:v[0-9]+]]
; GFX9: v_pk_lshlrev_b16 [[RESULT:v[0-9]+]], 8, [[LHS]]
define amdgpu_kernel void @shl_v_imm_v2i16(<2 x i16> addrspace(1)* %out, <2 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, <2 x i16> addrspace(1)* %out, i64 %tid.ext
  %vgpr = load <2 x i16>, <2 x i16> addrspace(1)* %in.gep
  %result = shl <2 x i16> %vgpr, <i16 8, i16 8>
  store <2 x i16> %result, <2 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_shl_v4i16:
; GCN: {{buffer|flat|global}}_load_dwordx2
; GCN: {{buffer|flat|global}}_load_dwordx2
; GFX9: v_pk_lshlrev_b16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_pk_lshlrev_b16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @v_shl_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %out, i64 %tid.ext
  %b_ptr = getelementptr <4 x i16>, <4 x i16> addrspace(1)* %in.gep, i32 1
  %a = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep
  %b = load <4 x i16>, <4 x i16> addrspace(1)* %b_ptr
  %result = shl <4 x i16> %a, %b
  store <4 x i16> %result, <4 x i16> addrspace(1)* %out.gep
  ret void
}

; GCN-LABEL: {{^}}shl_v_imm_v4i16:
; GCN: {{buffer|flat|global}}_load_dwordx2
; GFX9: v_pk_lshlrev_b16 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GFX9: v_pk_lshlrev_b16 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @shl_v_imm_v4i16(<4 x i16> addrspace(1)* %out, <4 x i16> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, <4 x i16> addrspace(1)* %out, i64 %tid.ext
  %vgpr = load <4 x i16>, <4 x i16> addrspace(1)* %in.gep
  %result = shl <4 x i16> %vgpr, <i16 8, i16 8, i16 8, i16 8>
  store <4 x i16> %result, <4 x i16> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
