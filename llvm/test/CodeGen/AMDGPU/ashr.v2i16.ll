; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=GFX9 %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=VI -check-prefix=CIVI %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=CI -check-prefix=CIVI %s

; FIXME: Should be same on CI/VI
; GCN-LABEL: {{^}}s_ashr_v2i16:
; GFX9: s_load_dword [[LHS:s[0-9]+]]
; GFX9: s_load_dword [[RHS:s[0-9]+]]
; GFX9: v_mov_b32_e32 [[VLHS:v[0-9]+]], [[LHS]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], [[RHS]], [[VLHS]]

; CIVI: s_load_dword [[LHS:s[0-9]+]]
; CIVI: s_load_dword [[RHS:s[0-9]+]]

; CIVI-DAG: s_ashr_i32
; CIVI-DAG: s_ashr_i32
; CIVI-DAG: s_sext_i32_i16
; CIVI-DAG: s_sext_i32_i16
; CIVI-DAG: s_ashr_i32
; CIVI-DAG: s_ashr_i32
; CIVI-DAG: s_lshl_b32
; CIVI: s_and_b32
; CIVI: s_or_b32

define amdgpu_kernel void @s_ashr_v2i16(ptr addrspace(1) %out, i32, <2 x i16> %lhs, i32, <2 x i16> %rhs) #0 {
  %result = ashr <2 x i16> %lhs, %rhs
  store <2 x i16> %result, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}v_ashr_v2i16:
; GCN: {{buffer|flat|global}}_load_dwordx2 v[[[LHS:[0-9]+]]:[[RHS:[0-9]+]]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], v[[RHS]], v[[LHS]]

; VI: v_ashrrev_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_ashrrev_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; CI-DAG: v_bfe_i32 v{{[0-9]+}}, v{{[0-9]+}}, 0, 16
; CI: v_ashrrev_i32_e32 v{{[0-9]+}}, 16, v[[LHS]]
; CI: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; CI: v_ashrrev_i32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_ashrrev_i32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; CI: v_lshlrev_b32_e32 v{{[0-9]+}}, 16, v{{[0-9]+}}
; CI: v_and_b32_e32 v{{[0-9]+}}, 0xffff, v{{[0-9]+}}
; CI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @v_ashr_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %b_ptr = getelementptr <2 x i16>, ptr addrspace(1) %in.gep, i32 1
  %a = load <2 x i16>, ptr addrspace(1) %in.gep
  %b = load <2 x i16>, ptr addrspace(1) %b_ptr
  %result = ashr <2 x i16> %a, %b
  store <2 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}ashr_v_s_v2i16:
; GFX9: s_load_dword [[RHS:s[0-9]+]]
; GFX9: {{buffer|flat|global}}_load_dword [[LHS:v[0-9]+]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], [[RHS]], [[LHS]]
define amdgpu_kernel void @ashr_v_s_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in, <2 x i16> %sgpr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %vgpr = load <2 x i16>, ptr addrspace(1) %in.gep
  %result = ashr <2 x i16> %vgpr, %sgpr
  store <2 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}ashr_s_v_v2i16:
; GFX9: s_load_dword [[LHS:s[0-9]+]]
; GFX9: {{buffer|flat|global}}_load_dword [[RHS:v[0-9]+]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], [[RHS]], [[LHS]]
define amdgpu_kernel void @ashr_s_v_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in, <2 x i16> %sgpr) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %vgpr = load <2 x i16>, ptr addrspace(1) %in.gep
  %result = ashr <2 x i16> %sgpr, %vgpr
  store <2 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}ashr_imm_v_v2i16:
; GCN: {{buffer|flat|global}}_load_dword [[RHS:v[0-9]+]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], [[RHS]], -4
define amdgpu_kernel void @ashr_imm_v_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %vgpr = load <2 x i16>, ptr addrspace(1) %in.gep
  %result = ashr <2 x i16> <i16 -4, i16 -4>, %vgpr
  store <2 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}ashr_v_imm_v2i16:
; GCN: {{buffer|flat|global}}_load_dword [[LHS:v[0-9]+]]
; GFX9: v_pk_ashrrev_i16 [[RESULT:v[0-9]+]], 8, [[LHS]]
define amdgpu_kernel void @ashr_v_imm_v2i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <2 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %vgpr = load <2 x i16>, ptr addrspace(1) %in.gep
  %result = ashr <2 x i16> %vgpr, <i16 8, i16 8>
  store <2 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}v_ashr_v4i16:
; GCN: {{buffer|flat|global}}_load_dwordx4
; GFX9: v_pk_ashrrev_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GFX9: v_pk_ashrrev_i16 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; VI: v_ashrrev_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_ashrrev_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_ashrrev_i16_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_ashrrev_i16_sdwa v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; VI: v_or_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}

; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @v_ashr_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %b_ptr = getelementptr <4 x i16>, ptr addrspace(1) %in.gep, i32 1
  %a = load <4 x i16>, ptr addrspace(1) %in.gep
  %b = load <4 x i16>, ptr addrspace(1) %b_ptr
  %result = ashr <4 x i16> %a, %b
  store <4 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

; GCN-LABEL: {{^}}ashr_v_imm_v4i16:
; GCN: {{buffer|flat|global}}_load_dwordx2
; GFX9: v_pk_ashrrev_i16 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GFX9: v_pk_ashrrev_i16 v{{[0-9]+}}, 8, v{{[0-9]+}}
; GCN: {{buffer|flat|global}}_store_dwordx2
define amdgpu_kernel void @ashr_v_imm_v4i16(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %tid.ext = sext i32 %tid to i64
  %in.gep = getelementptr inbounds <4 x i16>, ptr addrspace(1) %in, i64 %tid.ext
  %out.gep = getelementptr inbounds <4 x i16>, ptr addrspace(1) %out, i64 %tid.ext
  %vgpr = load <4 x i16>, ptr addrspace(1) %in.gep
  %result = ashr <4 x i16> %vgpr, <i16 8, i16 8, i16 8, i16 8>
  store <4 x i16> %result, ptr addrspace(1) %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
