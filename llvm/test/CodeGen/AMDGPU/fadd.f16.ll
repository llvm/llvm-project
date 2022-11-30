; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs -enable-misched=false < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=fiji -mattr=-flat-for-global -verify-machineinstrs -enable-misched=false < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}fadd_f16
; GCN: {{buffer|flat}}_load_ushort v[[A_F16:[0-9]+]]
; GCN: {{buffer|flat}}_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_add_f32_e32 v[[R_F32:[0-9]+]], v[[A_F32]], v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_add_f16_e32 v[[R_F16:[0-9]+]], v[[A_F16]], v[[B_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %a.val = load volatile half, ptr addrspace(1) %a
  %b.val = load volatile half, ptr addrspace(1) %b
  %r.val = fadd half %a.val, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fadd_f16_imm_a
; GCN: {{buffer|flat}}_load_ushort v[[B_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[B_F32:[0-9]+]], v[[B_F16]]
; SI:  v_add_f32_e32 v[[R_F32:[0-9]+]], 1.0, v[[B_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_add_f16_e32 v[[R_F16:[0-9]+]], 1.0, v[[B_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_f16_imm_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b) {
entry:
  %b.val = load half, ptr addrspace(1) %b
  %r.val = fadd half 1.0, %b.val
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fadd_f16_imm_b
; GCN: {{buffer|flat}}_load_ushort v[[A_F16:[0-9]+]]
; SI:  v_cvt_f32_f16_e32 v[[A_F32:[0-9]+]], v[[A_F16]]
; SI:  v_add_f32_e32 v[[R_F32:[0-9]+]], 2.0, v[[A_F32]]
; SI:  v_cvt_f16_f32_e32 v[[R_F16:[0-9]+]], v[[R_F32]]
; VI:  v_add_f16_e32 v[[R_F16:[0-9]+]], 2.0, v[[A_F16]]
; GCN: buffer_store_short v[[R_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_f16_imm_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %a.val = load half, ptr addrspace(1) %a
  %r.val = fadd half %a.val, 2.0
  store half %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fadd_v2f16:
; SI: buffer_load_dword v[[A_V2_F16:[0-9]+]]
; SI: buffer_load_dword v[[B_V2_F16:[0-9]+]]
; VI: flat_load_dword v[[A_V2_F16:[0-9]+]]
; VI: flat_load_dword v[[B_V2_F16:[0-9]+]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]

; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_0:[0-9]+]], v[[A_F32_0]], v[[B_F32_0]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_1:[0-9]+]], v[[A_F32_1]], v[[B_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI: v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI: v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_add_f16_e32 v[[R_F16_LO:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]]
; VI-DAG: v_add_f16_sdwa v[[R_F16_HI:[0-9]+]], v[[A_V2_F16]], v[[B_V2_F16]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_1
; VI:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_LO]], v[[R_F16_HI]]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_v2f16(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a,
    ptr addrspace(1) %b) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.a = getelementptr inbounds <2 x half>, ptr addrspace(1) %a, i32 %tid
  %gep.b = getelementptr inbounds <2 x half>, ptr addrspace(1) %b, i32 %tid
  %a.val = load <2 x half>, ptr addrspace(1) %gep.a
  %b.val = load <2 x half>, ptr addrspace(1) %gep.b
  %r.val = fadd <2 x half> %a.val, %b.val
  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fadd_v2f16_imm_a:
; GCN-DAG: {{buffer|flat}}_load_dword v[[B_V2_F16:[0-9]+]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_0:[0-9]+]], v[[B_V2_F16]]
; SI-DAG:  v_lshrrev_b32_e32 v[[B_F16_1:[0-9]+]], 16, v[[B_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[B_F32_1:[0-9]+]], v[[B_F16_1]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_0:[0-9]+]], 1.0, v[[B_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_1:[0-9]+]], 2.0, v[[B_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI:  v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_mov_b32_e32 v[[CONST2:[0-9]+]], 0x4000
; VI-DAG: v_add_f16_sdwa v[[R_F16_HI:[0-9]+]], v[[B_V2_F16]], v[[CONST2]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_f16_e32 v[[R_F16_0:[0-9]+]], 1.0, v[[B_V2_F16]]
; VI:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_v2f16_imm_a(
    ptr addrspace(1) %r,
    ptr addrspace(1) %b) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.b = getelementptr inbounds <2 x half>, ptr addrspace(1) %b, i32 %tid
  %b.val = load <2 x half>, ptr addrspace(1) %gep.b
  %r.val = fadd <2 x half> <half 1.0, half 2.0>, %b.val
  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

; GCN-LABEL: {{^}}fadd_v2f16_imm_b:
; GCN-DAG: {{buffer|flat}}_load_dword v[[A_V2_F16:[0-9]+]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_0:[0-9]+]], v[[A_V2_F16]]
; SI-DAG: v_lshrrev_b32_e32 v[[A_F16_1:[0-9]+]], 16, v[[A_V2_F16]]
; SI-DAG:  v_cvt_f32_f16_e32 v[[A_F32_1:[0-9]+]], v[[A_F16_1]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_0:[0-9]+]], 2.0, v[[A_F32_0]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_0:[0-9]+]], v[[R_F32_0]]
; SI-DAG:  v_add_f32_e32 v[[R_F32_1:[0-9]+]], 1.0, v[[A_F32_1]]
; SI-DAG:  v_cvt_f16_f32_e32 v[[R_F16_1:[0-9]+]], v[[R_F32_1]]
; SI-DAG:  v_lshlrev_b32_e32 v[[R_F16_HI:[0-9]+]], 16, v[[R_F16_1]]
; SI:  v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_0]], v[[R_F16_HI]]

; VI-DAG: v_mov_b32_e32 v[[CONST1:[0-9]+]], 0x3c00
; VI-DAG: v_add_f16_sdwa v[[R_F16_0:[0-9]+]], v[[A_V2_F16]], v[[CONST1]] dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:DWORD
; VI-DAG: v_add_f16_e32 v[[R_F16_1:[0-9]+]], 2.0, v[[A_V2_F16]]
; VI:     v_or_b32_e32 v[[R_V2_F16:[0-9]+]], v[[R_F16_1]], v[[R_F16_0]]

; GCN: buffer_store_dword v[[R_V2_F16]]
; GCN: s_endpgm
define amdgpu_kernel void @fadd_v2f16_imm_b(
    ptr addrspace(1) %r,
    ptr addrspace(1) %a) {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep.a = getelementptr inbounds <2 x half>, ptr addrspace(1) %a, i32 %tid
  %a.val = load <2 x half>, ptr addrspace(1) %gep.a
  %r.val = fadd <2 x half> %a.val, <half 2.0, half 1.0>
  store <2 x half> %r.val, ptr addrspace(1) %r
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
