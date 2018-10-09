; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s --check-prefix=CHECK-O0

; Test that we correctly legalize VGPR Rsrc operands in MUBUF instructions.

; CHECK-LABEL: mubuf_vgpr
; CHECK: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: [[LOOPBB:BB[0-9]+_[0-9]+]]:
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v0
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v1
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v2
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v3
; CHECK: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v[0:1]
; CHECK: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v[2:3]
; CHECK: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; CHECK: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_load_format_x [[RES:v[0-9]+]], v4, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; CHECK: s_xor_b64 exec, exec, [[CMP]]
; CHECK: s_cbranch_execnz [[LOOPBB]]
; CHECK: s_mov_b64 exec, [[SAVEEXEC]]
; CHECK: v_mov_b32_e32 v0, [[RES]]
define float @mubuf_vgpr(<4 x i32> %i, i32 %c) #0 {
  %call = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %i, i32 %c, i32 0, i1 zeroext false, i1 zeroext false) #1
  ret float %call
}

; CHECK-LABEL: mubuf_vgpr_adjacent_in_block

; CHECK: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v0
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v1
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v2
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v3
; CHECK: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v[0:1]
; CHECK: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v[2:3]
; CHECK: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; CHECK: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_load_format_x [[RES0:v[0-9]+]], v8, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; CHECK: s_xor_b64 exec, exec, [[CMP]]
; CHECK: s_cbranch_execnz [[LOOPBB0]]

; CHECK: s_mov_b64 exec, [[SAVEEXEC]]
; FIXME: redundant s_mov
; CHECK: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v4
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v5
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v6
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v7
; CHECK: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v[4:5]
; CHECK: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v[6:7]
; CHECK: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; CHECK: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_load_format_x [[RES1:v[0-9]+]], v8, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; CHECK: s_xor_b64 exec, exec, [[CMP]]
; CHECK: s_cbranch_execnz [[LOOPBB1]]

; CHECK: s_mov_b64 exec, [[SAVEEXEC]]
; CHECK-DAG: global_store_dword v[9:10], [[RES0]], off
; CHECK-DAG: global_store_dword v[11:12], [[RES1]], off

define void @mubuf_vgpr_adjacent_in_block(<4 x i32> %i, <4 x i32> %j, i32 %c, float addrspace(1)* %out0, float addrspace(1)* %out1) #0 {
entry:
  %val0 = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %i, i32 %c, i32 0, i1 zeroext false, i1 zeroext false) #1
  %val1 = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %j, i32 %c, i32 0, i1 zeroext false, i1 zeroext false) #1
  store volatile float %val0, float addrspace(1)* %out0
  store volatile float %val1, float addrspace(1)* %out1
  ret void
}

; CHECK-LABEL: mubuf_vgpr_outside_entry

; CHECK-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s4
; CHECK-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v0
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v1
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v2
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v3
; CHECK: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v[0:1]
; CHECK: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v[2:3]
; CHECK: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; CHECK: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; CHECK: s_xor_b64 exec, exec, [[CMP]]
; CHECK: s_cbranch_execnz [[LOOPBB0]]

; CHECK: s_mov_b64 exec, [[SAVEEXEC]]
; CHECK: s_cbranch_execz [[TERMBB:BB[0-9]+_[0-9]+]]

; CHECK: BB{{[0-9]+_[0-9]+}}:
; CHECK-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s4
; CHECK-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; CHECK: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v4
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v5
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v6
; CHECK-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v7
; CHECK: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v[4:5]
; CHECK: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v[6:7]
; CHECK: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; CHECK: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK: s_waitcnt vmcnt(0)
; CHECK: buffer_load_format_x [[RES]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; CHECK: s_xor_b64 exec, exec, [[CMP]]
; CHECK: s_cbranch_execnz [[LOOPBB1]]

; CHECK: s_mov_b64 exec, [[SAVEEXEC]]

; CHECK: [[TERMBB]]:
; CHECK: global_store_dword v[11:12], [[RES]], off

; Confirm spills do not occur between the XOR and branch that terminate the
; waterfall loop BBs.

; CHECK-O0-LABEL: mubuf_vgpr_outside_entry

; CHECK-O0-DAG: s_mov_b32 [[IDX_S:s[0-9]+]], s4
; CHECK-O0-DAG: v_mov_b32_e32 [[IDX_V:v[0-9]+]], [[IDX_S]]
; CHECK-O0-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; CHECK-O0-DAG: buffer_store_dword [[IDX_V]], off, s[0:3], s5 offset:[[IDX_OFF:[0-9]+]] ; 4-byte Folded Spill

; CHECK-O0: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; CHECK-O0: buffer_load_dword v[[VRSRC0:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC1:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC2:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC3:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP0:[0-9]+]], v[[VRSRC0]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP1:[0-9]+]], v[[VRSRC1]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP2:[0-9]+]], v[[VRSRC2]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP3:[0-9]+]], v[[VRSRC3]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC0:[0-9]+]], s[[SRSRCTMP0]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC1:[0-9]+]], s[[SRSRCTMP1]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC2:[0-9]+]], s[[SRSRCTMP2]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC3:[0-9]+]], s[[SRSRCTMP3]]
; CHECK-O0: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; CHECK-O0: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; CHECK-O0: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; CHECK-O0: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK-O0: buffer_load_dword [[IDX:v[0-9]+]], off, s[0:3], s5 offset:[[IDX_OFF]] ; 4-byte Folded Reload
; CHECK-O0: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, {{.*}} idxen
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_store_dword [[RES]], off, s[0:3], s5 offset:[[RES_OFF_TMP:[0-9]+]] ; 4-byte Folded Spill
; CHECK-O0: s_xor_b64 exec, exec, [[CMP]]
; CHECK-O0-NEXT: s_cbranch_execnz [[LOOPBB0]]

; CHECK-O0: s_mov_b64 exec, [[SAVEEXEC]]
; CHECK-O0: buffer_load_dword [[RES:v[0-9]+]], off, s[0:3], s5 offset:[[RES_OFF_TMP]] ; 4-byte Folded Reload
; CHECK-O0: buffer_store_dword [[RES]], off, s[0:3], s5 offset:[[RES_OFF:[0-9]+]] ; 4-byte Folded Spill
; CHECK-O0: s_cbranch_execz [[TERMBB:BB[0-9]+_[0-9]+]]

; CHECK-O0: BB{{[0-9]+_[0-9]+}}:
; CHECK-O0-DAG: s_mov_b64 s{{\[}}[[SAVEEXEC0:[0-9]+]]:[[SAVEEXEC1:[0-9]+]]{{\]}}, exec
; CHECK-O0-DAG: buffer_store_dword {{v[0-9]+}}, off, s[0:3], s5 offset:[[IDX_OFF:[0-9]+]] ; 4-byte Folded Spill
; CHECK-O0: v_writelane_b32 [[VSAVEEXEC:v[0-9]+]], s[[SAVEEXEC0]], [[SAVEEXEC_IDX0:[0-9]+]]
; CHECK-O0: v_writelane_b32 [[VSAVEEXEC:v[0-9]+]], s[[SAVEEXEC1]], [[SAVEEXEC_IDX1:[0-9]+]]

; CHECK-O0: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; CHECK-O0: buffer_load_dword v[[VRSRC0:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC1:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC2:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_load_dword v[[VRSRC3:[0-9]+]], {{.*}} ; 4-byte Folded Reload
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP0:[0-9]+]], v[[VRSRC0]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP1:[0-9]+]], v[[VRSRC1]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP2:[0-9]+]], v[[VRSRC2]]
; CHECK-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP3:[0-9]+]], v[[VRSRC3]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC0:[0-9]+]], s[[SRSRCTMP0]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC1:[0-9]+]], s[[SRSRCTMP1]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC2:[0-9]+]], s[[SRSRCTMP2]]
; CHECK-O0-DAG: s_mov_b32 s[[SRSRC3:[0-9]+]], s[[SRSRCTMP3]]
; CHECK-O0: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; CHECK-O0: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; CHECK-O0: s_and_b64 [[CMP:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; CHECK-O0: s_and_saveexec_b64 [[CMP]], [[CMP]]
; CHECK-O0: buffer_load_dword [[IDX:v[0-9]+]], off, s[0:3], s5 offset:[[IDX_OFF]] ; 4-byte Folded Reload
; CHECK-O0: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, {{.*}} idxen
; CHECK-O0: s_waitcnt vmcnt(0)
; CHECK-O0: buffer_store_dword [[RES]], off, s[0:3], s5 offset:[[RES_OFF_TMP:[0-9]+]] ; 4-byte Folded Spill
; CHECK-O0: s_xor_b64 exec, exec, [[CMP]]
; CHECK-O0-NEXT: s_cbranch_execnz [[LOOPBB1]]

; CHECK-O0: v_readlane_b32 s[[SAVEEXEC0:[0-9]+]], [[VSAVEEXEC]], [[SAVEEXEC_IDX0]]
; CHECK-O0: v_readlane_b32 s[[SAVEEXEC1:[0-9]+]], [[VSAVEEXEC]], [[SAVEEXEC_IDX1]]
; CHECK-O0: s_mov_b64 exec, s{{\[}}[[SAVEEXEC0]]:[[SAVEEXEC1]]{{\]}}
; CHECK-O0: buffer_load_dword [[RES:v[0-9]+]], off, s[0:3], s5 offset:[[RES_OFF_TMP]] ; 4-byte Folded Reload
; CHECK-O0: buffer_store_dword [[RES]], off, s[0:3], s5 offset:[[RES_OFF]] ; 4-byte Folded Spill

; CHECK-O0: [[TERMBB]]:
; CHECK-O0: buffer_load_dword [[RES:v[0-9]+]], off, s[0:3], s5 offset:[[RES_OFF]] ; 4-byte Folded Reload
; CHECK-O0: global_store_dword v[{{[0-9]+:[0-9]+}}], [[RES]], off

define void @mubuf_vgpr_outside_entry(<4 x i32> %i, <4 x i32> %j, i32 %c, float addrspace(1)* %in, float addrspace(1)* %out) #0 {
entry:
  %live.out.reg = call i32 asm sideeffect "s_mov_b32 $0, 17", "={s4}" ()
  %val0 = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %i, i32 %live.out.reg, i32 0, i1 zeroext false, i1 zeroext false) #1
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %cmp = icmp eq i32 %idx, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  %val1 = call float @llvm.amdgcn.buffer.load.format.f32(<4 x i32> %j, i32 %live.out.reg, i32 0, i1 zeroext false, i1 zeroext false) #1
  br label %bb2

bb2:
  %val = phi float [ %val0, %entry ], [ %val1, %bb1 ]
  store volatile float %val, float addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.amdgcn.buffer.load.format.f32(<4 x i32>, i32, i32, i1, i1) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
