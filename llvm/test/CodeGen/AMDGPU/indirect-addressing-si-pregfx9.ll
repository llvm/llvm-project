; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL,PREGFX9 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,MOVREL,PREGFX9 %s
; RUN: llc -amdgpu-scalarize-global-loads=false -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -amdgpu-vgpr-index-mode -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,IDXMODE,PREGFX9 %s

; Tests for indirect addressing on SI, which is implemented using dynamic
; indexing of vectors.

; Subtest below moved from file test/CodeGen/AMDGPU/indirect-addressing-si.ll
; to avoid gfx9 scheduling induced issues.


; GCN-LABEL: {{^}}insert_vgpr_offset_multiple_in_block:
; GCN-DAG: s_load_dwordx16 s{{\[}}[[S_ELT0:[0-9]+]]:[[S_ELT15:[0-9]+]]{{\]}}
; GCN-DAG: {{buffer|flat|global}}_load_dword [[IDX0:v[0-9]+]]
; GCN-DAG: v_mov_b32 [[INS0:v[0-9]+]], 62

; GCN-DAG: v_mov_b32_e32 v[[VEC_ELT15:[0-9]+]], s[[S_ELT15]]
; GCN-DAG: v_mov_b32_e32 v[[VEC_ELT0:[0-9]+]], s[[S_ELT0]]

; GCN-DAG: v_add_{{i32|u32}}_e32 [[IDX1:v[0-9]+]], vcc, 1, [[IDX0]]

; GCN: [[LOOP0:BB[0-9]+_[0-9]+]]:
; GCN-NEXT: v_readfirstlane_b32 [[READLANE:s[0-9]+]], [[IDX0]]
; GCN: v_cmp_eq_u32_e32 vcc, [[READLANE]], [[IDX0]]
; GCN: s_and_saveexec_b64 vcc, vcc

; MOVREL: s_mov_b32 m0, [[READLANE]]
; MOVREL-NEXT: v_movreld_b32_e32 v[[VEC_ELT0]], [[INS0]]

; IDXMODE: s_set_gpr_idx_on [[READLANE]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 v[[VEC_ELT0]], [[INS0]]
; IDXMODE: s_set_gpr_idx_off

; GCN-NEXT: s_xor_b64 exec, exec, vcc
; GCN: s_cbranch_execnz [[LOOP0]]

; FIXME: Redundant copy
; GCN: s_mov_b64 exec, [[MASK:s\[[0-9]+:[0-9]+\]]]

; GCN: s_mov_b64 [[MASK]], exec

; GCN: [[LOOP1:BB[0-9]+_[0-9]+]]:
; GCN-NEXT: v_readfirstlane_b32 [[READLANE:s[0-9]+]], [[IDX1]]
; GCN: v_cmp_eq_u32_e32 vcc, [[READLANE]], [[IDX1]]
; GCN: s_and_saveexec_b64 vcc, vcc

; MOVREL: s_mov_b32 m0, [[READLANE]]
; MOVREL-NEXT: v_movreld_b32_e32 v{{[0-9]+}}, 63

; IDXMODE: s_set_gpr_idx_on [[READLANE]], gpr_idx(DST)
; IDXMODE-NEXT: v_mov_b32_e32 v{{[0-9]+}}, 63
; IDXMODE: s_set_gpr_idx_off

; GCN-NEXT: s_xor_b64 exec, exec, vcc
; GCN: s_cbranch_execnz [[LOOP1]]

; GCN: buffer_store_dwordx4 v{{\[}}[[VEC_ELT0]]:

; GCN: buffer_store_dword [[INS0]]
define amdgpu_kernel void @insert_vgpr_offset_multiple_in_block(<16 x i32> addrspace(1)* %out0, <16 x i32> addrspace(1)* %out1, i32 addrspace(1)* %in, <16 x i32> %vec0) #0 {
entry:
  %id = call i32 @llvm.amdgcn.workitem.id.x() #1
  %id.ext = zext i32 %id to i64
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i64 %id.ext
  %idx0 = load volatile i32, i32 addrspace(1)* %gep
  %idx1 = add i32 %idx0, 1
  %live.out.val = call i32 asm sideeffect "v_mov_b32 $0, 62", "=v"()
  %vec1 = insertelement <16 x i32> %vec0, i32 %live.out.val, i32 %idx0
  %vec2 = insertelement <16 x i32> %vec1, i32 63, i32 %idx1
  store volatile <16 x i32> %vec2, <16 x i32> addrspace(1)* %out0
  %cmp = icmp eq i32 %id, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  store volatile i32 %live.out.val, i32 addrspace(1)* undef
  br label %bb2

bb2:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind convergent }
