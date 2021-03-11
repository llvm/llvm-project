; RUN: llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s --check-prefix=W64
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=+wavefrontsize32,-wavefrontsize64 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s --check-prefix=W32
; RUN: llc -march=amdgcn -mcpu=gfx1010 -mattr=-wavefrontsize32,+wavefrontsize64 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s --check-prefix=W64
; RUN: llc -O0 -march=amdgcn -mcpu=gfx900 -verify-machineinstrs -verify-machine-dom-info -o - %s | FileCheck %s --check-prefix=W64-O0

; Test that we correctly legalize VGPR Rsrc operands in MUBUF instructions.

; W64-LABEL: mubuf_vgpr
; W64: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; W64: [[LOOPBB:BB[0-9]+_[0-9]+]]:
; W64-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W64: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; W64: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64: buffer_load_format_x [[RES:v[0-9]+]], v{{[0-9]+}}, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W64: s_xor_b64 exec, exec, [[AND]]
; W64: s_cbranch_execnz [[LOOPBB]]
; W64: s_mov_b64 exec, [[SAVEEXEC]]
; W64: v_mov_b32_e32 v0, [[RES]]

; W32-LABEL: mubuf_vgpr
; W32: s_mov_b32 [[SAVEEXEC:s[0-9]+]], exec_lo
; W32: [[LOOPBB:BB[0-9]+_[0-9]+]]:
; W32-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W32: v_cmp_eq_u64_e32 vcc_lo, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W32: v_cmp_eq_u64_e64 [[CMP0:s[0-9]+]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W32: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[CMP0]]
; W32: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[AND]]
; W32: buffer_load_format_x [[RES:v[0-9]+]], v{{[0-9]+}}, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W32: s_xor_b32 exec_lo, exec_lo, [[SAVE]]
; W32: s_cbranch_execnz [[LOOPBB]]
; W32: s_mov_b32 exec_lo, [[SAVEEXEC]]
; W32: v_mov_b32_e32 v0, [[RES]]

define float @mubuf_vgpr(<4 x i32> %i, i32 %c) #0 {
  %call = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %i, i32 %c, i32 0, i32 0, i32 0) #1
  ret float %call
}


; W64-LABEL: mubuf_vgpr_adjacent_in_block

; W64: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; W64: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; W64-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W64: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; W64: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64: buffer_load_format_x [[RES0:v[0-9]+]], v{{[0-9]+}}, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W64: s_xor_b64 exec, exec, [[SAVE]]
; W64: s_cbranch_execnz [[LOOPBB0]]

; W64: s_mov_b64 exec, [[SAVEEXEC]]
; FIXME: redundant s_mov
; W64: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; W64: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; W64-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W64: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; W64: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64: buffer_load_format_x [[RES1:v[0-9]+]], v{{[0-9]+}}, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W64: s_xor_b64 exec, exec, [[SAVE]]
; W64: s_cbranch_execnz [[LOOPBB1]]

; W64: s_mov_b64 exec, [[SAVEEXEC]]
; W64-DAG: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES0]], off
; W64-DAG: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES1]], off


; W32-LABEL: mubuf_vgpr_adjacent_in_block

; W32: s_mov_b32 [[SAVEEXEC:s[0-9]+]], exec_lo
; W32: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; W32-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W32: v_cmp_eq_u64_e32 vcc_lo, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W32: v_cmp_eq_u64_e64 [[CMP0:s[0-9]+]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W32: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[CMP0]]
; W32: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[AND]]
; W32: buffer_load_format_x [[RES0:v[0-9]+]], v{{[0-9]+}}, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W32: s_xor_b32 exec_lo, exec_lo, [[SAVE]]
; W32: s_cbranch_execnz [[LOOPBB0]]

; W32: s_mov_b32 exec_lo, [[SAVEEXEC]]
; FIXME: redundant s_mov
; W32: s_mov_b32 [[SAVEEXEC:s[0-9]+]], exec_lo

; W32: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; W32-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W32: v_cmp_eq_u64_e32 vcc_lo, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W32: v_cmp_eq_u64_e64 [[CMP0:s[0-9]+]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W32: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[CMP0]]
; W32: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[AND]]
; W32: buffer_load_format_x [[RES1:v[0-9]+]], v8, s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W32: s_xor_b32 exec_lo, exec_lo, [[SAVE]]
; W32: s_cbranch_execnz [[LOOPBB1]]

; W32: s_mov_b32 exec_lo, [[SAVEEXEC]]
; W32-DAG: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES0]], off
; W32-DAG: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES1]], off

define void @mubuf_vgpr_adjacent_in_block(<4 x i32> %i, <4 x i32> %j, i32 %c, float addrspace(1)* %out0, float addrspace(1)* %out1) #0 {
entry:
  %val0 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %i, i32 %c, i32 0, i32 0, i32 0) #1
  %val1 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %j, i32 %c, i32 0, i32 0, i32 0) #1
  store volatile float %val0, float addrspace(1)* %out0
  store volatile float %val1, float addrspace(1)* %out1
  ret void
}


; W64-LABEL: mubuf_vgpr_outside_entry

; W64-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s{{[0-9]+}}
; W64-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; W64: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; W64-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W64: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; W64: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W64: s_xor_b64 exec, exec, [[SAVE]]
; W64: s_cbranch_execnz [[LOOPBB0]]

; W64: s_mov_b64 exec, [[SAVEEXEC]]
; W64: s_cbranch_execz [[TERMBB:BB[0-9]+_[0-9]+]]

; W64: ; %bb.{{[0-9]+}}:
; W64-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s{{[0-9]+}}
; W64-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec

; W64: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; W64-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W64-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W64: v_cmp_eq_u64_e32 vcc, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], vcc, [[CMP0]]
; W64: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64: buffer_load_format_x [[RES]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W64: s_xor_b64 exec, exec, [[SAVE]]
; W64: s_cbranch_execnz [[LOOPBB1]]

; W64: s_mov_b64 exec, [[SAVEEXEC]]

; W64: [[TERMBB]]:
; W64: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]], off


; W32-LABEL: mubuf_vgpr_outside_entry

; W32-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s4
; W32-DAG: s_mov_b32 [[SAVEEXEC:s[0-9]+]], exec_lo

; W32: [[LOOPBB0:BB[0-9]+_[0-9]+]]:
; W32-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W32: v_cmp_eq_u64_e32 vcc_lo, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W32: v_cmp_eq_u64_e64 [[CMP0:s[0-9]+]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W32: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[CMP0]]
; W32: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[AND]]
; W32: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W32: s_xor_b32 exec_lo, exec_lo, [[SAVE]]
; W32: s_cbranch_execnz [[LOOPBB0]]

; W32: s_mov_b32 exec_lo, [[SAVEEXEC]]
; W32: s_cbranch_execz [[TERMBB:BB[0-9]+_[0-9]+]]

; W32: ; %bb.{{[0-9]+}}:
; W32-DAG: v_mov_b32_e32 [[IDX:v[0-9]+]], s4
; W32-DAG: s_mov_b32 [[SAVEEXEC:s[0-9]+]], exec_lo

; W32: [[LOOPBB1:BB[0-9]+_[0-9]+]]:
; W32-DAG: v_readfirstlane_b32 s[[SRSRC0:[0-9]+]], v[[VRSRC0:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC1:[0-9]+]], v[[VRSRC1:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC2:[0-9]+]], v[[VRSRC2:[0-9]+]]
; W32-DAG: v_readfirstlane_b32 s[[SRSRC3:[0-9]+]], v[[VRSRC3:[0-9]+]]
; W32: v_cmp_eq_u64_e32 vcc_lo, s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W32: v_cmp_eq_u64_e64 [[CMP0:s[0-9]+]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W32: s_and_b32 [[AND:s[0-9]+]], vcc_lo, [[CMP0]]
; W32: s_and_saveexec_b32 [[SAVE:s[0-9]+]], [[AND]]
; W32: buffer_load_format_x [[RES]], [[IDX]], s{{\[}}[[SRSRC0]]:[[SRSRC3]]{{\]}}, 0 idxen
; W32: s_xor_b32 exec_lo, exec_lo, [[SAVE]]
; W32: s_cbranch_execnz [[LOOPBB1]]

; W32: s_mov_b32 exec_lo, [[SAVEEXEC]]

; W32: [[TERMBB]]:
; W32: global_store_dword v{{\[[0-9]+:[0-9]+\]}}, [[RES]], off


; Confirm spills do not occur between the XOR and branch that terminate the
; waterfall loop BBs.

; W64-O0-LABEL: mubuf_vgpr_outside_entry

; W64-O0-DAG: s_mov_b32 [[IDX_S:s[0-9]+]], s{{[0-9]+}}
; W64-O0-DAG: v_mov_b32_e32 [[IDX_V:v[0-9]+]], s{{[0-9]+}}
; W64-O0-DAG: s_mov_b64 [[SAVEEXEC:s\[[0-9]+:[0-9]+\]]], exec
; W64-O0-DAG: buffer_store_dword [[IDX_V]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} ; 4-byte Folded Spill

; W64-O0: [[LOOPBB0:BB[0-9]+_[0-9]+]]: ; =>This Inner Loop Header: Depth=1
; W64-O0: buffer_load_dword [[IDX:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s32 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC0:[0-9]+]], off, s[0:3], s32 offset:28 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC1:[0-9]+]], off, s[0:3], s32 offset:32 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC2:[0-9]+]], off, s[0:3], s32 offset:36 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC3:[0-9]+]], off, s[0:3], s32 offset:40 ; 4-byte Folded Reload
; W64-O0: s_waitcnt vmcnt(0)
; W64-O0-DAG: v_readfirstlane_b32 s[[S0:[0-9]+]], v[[VRSRC0]]
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP1:[0-9]+]], v[[VRSRC1]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC0:[0-9]+]], s[[S0]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC1:[0-9]+]], s[[SRSRCTMP1]]
; W64-O0-DAG: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP2:[0-9]+]], v[[VRSRC2]]
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP3:[0-9]+]], v[[VRSRC3]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC2:[0-9]+]], s[[SRSRCTMP2]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC3:[0-9]+]], s[[SRSRCTMP3]]
; W64-O0-DAG: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64-O0-DAG: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; W64-O0-DAG: s_mov_b32 s[[S1:[0-9]+]], s[[SRSRCTMP1]]
; W64-O0-DAG: s_mov_b32 s[[S2:[0-9]+]], s[[SRSRCTMP2]]
; W64-O0-DAG: s_mov_b32 s[[S3:[0-9]+]], s[[SRSRCTMP3]]
; W64-O0: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64-O0: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[S0]]:[[S3]]{{\]}}, {{.*}} idxen
; W64-O0: s_waitcnt vmcnt(0)
; W64-O0: buffer_store_dword [[RES]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF_TMP:[0-9]+]] ; 4-byte Folded Spill
; W64-O0: s_xor_b64 exec, exec, [[SAVE]]
; W64-O0-NEXT: s_cbranch_execnz [[LOOPBB0]]

; XXX-W64-O0: s_mov_b64 exec, [[SAVEEXEC]]
; W64-O0: buffer_load_dword [[RES:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF_TMP]] ; 4-byte Folded Reload
; W64-O0: buffer_store_dword [[RES]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF:[0-9]+]] ; 4-byte Folded Spill
; W64-O0: s_cbranch_execz [[TERMBB:BB[0-9]+_[0-9]+]]

; W64-O0: ; %bb.{{[0-9]+}}: ; %bb1
; W64-O0-DAG: buffer_store_dword {{v[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, s32 offset:[[IDX_OFF:[0-9]+]] ; 4-byte Folded Spill
; W64-O0-DAG: s_mov_b64 s{{\[}}[[SAVEEXEC0:[0-9]+]]:[[SAVEEXEC1:[0-9]+]]{{\]}}, exec
; W64-O0: v_writelane_b32 [[VSAVEEXEC:v[0-9]+]], s[[SAVEEXEC0]], [[SAVEEXEC_IDX0:[0-9]+]]
; W64-O0: v_writelane_b32 [[VSAVEEXEC]], s[[SAVEEXEC1]], [[SAVEEXEC_IDX1:[0-9]+]]

; W64-O0: [[LOOPBB1:BB[0-9]+_[0-9]+]]: ; =>This Inner Loop Header: Depth=1
; W64-O0: buffer_load_dword [[IDX:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s32  offset:[[IDX_OFF]] ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC0:[0-9]+]], off, s[0:3], s32 offset:4 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC1:[0-9]+]], off, s[0:3], s32 offset:8 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC2:[0-9]+]], off, s[0:3], s32 offset:12 ; 4-byte Folded Reload
; W64-O0: buffer_load_dword v[[VRSRC3:[0-9]+]], off, s[0:3], s32 offset:16 ; 4-byte Folded Reload
; W64-O0: s_waitcnt vmcnt(0)
; W64-O0-DAG: v_readfirstlane_b32 s[[S0:[0-9]+]], v[[VRSRC0]]
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP1:[0-9]+]], v[[VRSRC1]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC0:[0-9]+]], s[[S0]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC1:[0-9]+]], s[[SRSRCTMP1]]
; W64-O0-DAG: v_cmp_eq_u64_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC0]]:[[SRSRC1]]{{\]}}, v{{\[}}[[VRSRC0]]:[[VRSRC1]]{{\]}}
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP2:[0-9]+]], v[[VRSRC2]]
; W64-O0-DAG: v_readfirstlane_b32 s[[SRSRCTMP3:[0-9]+]], v[[VRSRC3]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC2:[0-9]+]], s[[SRSRCTMP2]]
; W64-O0-DAG: s_mov_b32 s[[SRSRC3:[0-9]+]], s[[SRSRCTMP3]]
; W64-O0-DAG: v_cmp_eq_u64_e64 [[CMP1:s\[[0-9]+:[0-9]+\]]], s{{\[}}[[SRSRC2]]:[[SRSRC3]]{{\]}}, v{{\[}}[[VRSRC2]]:[[VRSRC3]]{{\]}}
; W64-O0-DAG: s_and_b64 [[AND:s\[[0-9]+:[0-9]+\]]], [[CMP0]], [[CMP1]]
; W64-O0-DAG: s_mov_b32 s[[S1:[0-9]+]], s[[SRSRCTMP1]]
; W64-O0-DAG: s_mov_b32 s[[S2:[0-9]+]], s[[SRSRCTMP2]]
; W64-O0-DAG: s_mov_b32 s[[S3:[0-9]+]], s[[SRSRCTMP3]]
; W64-O0: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[AND]]
; W64-O0: buffer_load_format_x [[RES:v[0-9]+]], [[IDX]], s{{\[}}[[S0]]:[[S3]]{{\]}}, {{.*}} idxen
; W64-O0: s_waitcnt vmcnt(0)
; W64-O0: buffer_store_dword [[RES]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF_TMP:[0-9]+]] ; 4-byte Folded Spill
; W64-O0: s_xor_b64 exec, exec, [[SAVE]]
; W64-O0-NEXT: s_cbranch_execnz [[LOOPBB1]]

; W64-O0: v_readlane_b32 s[[SAVEEXEC0:[0-9]+]], [[VSAVEEXEC]], [[SAVEEXEC_IDX0]]
; W64-O0: v_readlane_b32 s[[SAVEEXEC1:[0-9]+]], [[VSAVEEXEC]], [[SAVEEXEC_IDX1]]
; W64-O0: s_mov_b64 exec, s{{\[}}[[SAVEEXEC0]]:[[SAVEEXEC1]]{{\]}}
; W64-O0: buffer_load_dword [[RES:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF_TMP]] ; 4-byte Folded Reload
; W64-O0: buffer_store_dword [[RES]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF]] ; 4-byte Folded Spill

; W64-O0: [[TERMBB]]:
; W64-O0: buffer_load_dword [[RES:v[0-9]+]], off, s{{\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}} offset:[[RES_OFF]] ; 4-byte Folded Reload
; W64-O0: global_store_dword v[{{[0-9]+:[0-9]+}}], [[RES]], off

define void @mubuf_vgpr_outside_entry(<4 x i32> %i, <4 x i32> %j, i32 %c, float addrspace(1)* %in, float addrspace(1)* %out) #0 {
entry:
  %live.out.reg = call i32 asm sideeffect "s_mov_b32 $0, 17", "={s4}" ()
  %val0 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %i, i32 %live.out.reg, i32 0, i32 0, i32 0) #1
  %idx = call i32 @llvm.amdgcn.workitem.id.x() #1
  %cmp = icmp eq i32 %idx, 0
  br i1 %cmp, label %bb1, label %bb2

bb1:
  %val1 = call float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32> %j, i32 %live.out.reg, i32 0, i32 0, i32 0) #1
  br label %bb2

bb2:
  %val = phi float [ %val0, %entry ], [ %val1, %bb1 ]
  store volatile float %val, float addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.amdgcn.struct.buffer.load.format.f32(<4 x i32>, i32, i32, i32, i32 immarg) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
