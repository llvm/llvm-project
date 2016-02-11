; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; XUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

; FIXME: Enable VI

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare float @llvm.fabs.f32(float) nounwind readnone

; GCN-LABEL: {{^}}madak_f32:
; GCN: buffer_load_dword [[VA:v[0-9]+]]
; GCN: buffer_load_dword [[VB:v[0-9]+]]
; GCN: v_madak_f32_e32 {{v[0-9]+}}, [[VA]], [[VB]], 0x41200000
define void @madak_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; Make sure this is only folded with one use. This is a code size
; optimization and if we fold the immediate multiple times, we'll undo
; it.

; GCN-LABEL: {{^}}madak_2_use_f32:
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; GCN-DAG: buffer_load_dword [[VB:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; GCN-DAG: buffer_load_dword [[VC:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GCN-DAG: v_mad_f32 {{v[0-9]+}}, [[VB]], [[VA]], [[VK]]
; GCN-DAG: v_mac_f32_e32 [[VK]], [[VC]], [[VA]]
; GCN: s_endpgm
define void @madak_2_use_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

  %in.gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %in.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1
  %in.gep.2 = getelementptr float, float addrspace(1)* %in.gep.0, i32 2

  %out.gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %out.gep.1 = getelementptr float, float addrspace(1)* %in.gep.0, i32 1

  %a = load float, float addrspace(1)* %in.gep.0, align 4
  %b = load float, float addrspace(1)* %in.gep.1, align 4
  %c = load float, float addrspace(1)* %in.gep.2, align 4

  %mul0 = fmul float %a, %b
  %mul1 = fmul float %a, %c
  %madak0 = fadd float %mul0, 10.0
  %madak1 = fadd float %mul1, 10.0

  store float %madak0, float addrspace(1)* %out.gep.0, align 4
  store float %madak1, float addrspace(1)* %out.gep.1, align 4
  ret void
}

; GCN-LABEL: {{^}}madak_m_inline_imm_f32:
; GCN: buffer_load_dword [[VA:v[0-9]+]]
; GCN: v_madak_f32_e32 {{v[0-9]+}}, 4.0, [[VA]], 0x41200000
define void @madak_m_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4

  %mul = fmul float 4.0, %a
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; Make sure nothing weird happens with a value that is also allowed as
; an inline immediate.

; GCN-LABEL: {{^}}madak_inline_imm_f32:
; GCN: buffer_load_dword [[VA:v[0-9]+]]
; GCN: buffer_load_dword [[VB:v[0-9]+]]
; GCN: v_mad_f32 {{v[0-9]+}}, [[VA]], [[VB]], 4.0
define void @madak_inline_imm_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 4.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; We can't use an SGPR when forming madak
; GCN-LABEL: {{^}}s_v_madak_f32:
; GCN: s_load_dword [[SB:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]]
; GCN-NOT: v_madak_f32
; GCN: v_mac_f32_e32 [[VK]], [[SB]], [[VA]]
define void @s_v_madak_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float %b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: @v_s_madak_f32
; GCN-DAG: s_load_dword [[SB:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x41200000
; GCN-DAG: buffer_load_dword [[VA:v[0-9]+]]
; GCN-NOT: v_madak_f32
; GCN: v_mac_f32_e32 [[VK]], [[SB]], [[VA]]
define void @v_s_madak_f32(float addrspace(1)* noalias %out, float %a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}s_s_madak_f32:
; GCN-NOT: v_madak_f32
; GCN: v_mac_f32_e32 {{v[0-9]+}}, {{s[0-9]+}}, {{v[0-9]+}}
define void @s_s_madak_f32(float addrspace(1)* %out, float %a, float %b) nounwind {
  %mul = fmul float %a, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madak_src0_modifier_f32:
; GCN: buffer_load_dword [[VA:v[0-9]+]]
; GCN: buffer_load_dword [[VB:v[0-9]+]]
; GCN: v_mad_f32 {{v[0-9]+}}, |{{v[0-9]+}}|, {{v[0-9]+}}, {{[sv][0-9]+}}
; GCN: s_endpgm
define void @no_madak_src0_modifier_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %a.fabs = call float @llvm.fabs.f32(float %a) nounwind readnone

  %mul = fmul float %a.fabs, %b
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}

; GCN-LABEL: {{^}}no_madak_src1_modifier_f32:
; GCN: buffer_load_dword [[VA:v[0-9]+]]
; GCN: buffer_load_dword [[VB:v[0-9]+]]
; GCN: v_mad_f32 {{v[0-9]+}}, {{v[0-9]+}}, |{{v[0-9]+}}|, {{[sv][0-9]+}}
; GCN: s_endpgm
define void @no_madak_src1_modifier_f32(float addrspace(1)* noalias %out, float addrspace(1)* noalias %in.a, float addrspace(1)* noalias %in.b) nounwind {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %in.a.gep = getelementptr float, float addrspace(1)* %in.a, i32 %tid
  %in.b.gep = getelementptr float, float addrspace(1)* %in.b, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid

  %a = load float, float addrspace(1)* %in.a.gep, align 4
  %b = load float, float addrspace(1)* %in.b.gep, align 4

  %b.fabs = call float @llvm.fabs.f32(float %b) nounwind readnone

  %mul = fmul float %a, %b.fabs
  %madak = fadd float %mul, 10.0
  store float %madak, float addrspace(1)* %out.gep, align 4
  ret void
}
